import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import kornia
from imageio import imwrite
import json
from .BaseTrainer import BaseLitModule




def transform_anchor_points(A, *argv):
    """
    perform matrix multiplication A*anchor_point for each body part and anchor_point
    Args:
        A: torch.tensor (batch, num_parts, 3, 3) of transformation matrices
        *args: tensors with shape (batch, num_parts, num_anchors, 3)
    """

    num_parts = 0
    for arg in argv:
        num_parts += arg.shape[1]
    assert num_parts == A.shape[1], "number of matrices should match number of parts!"

    index = 0
    transformed = []
    for arg in argv:
        num_parts = arg.shape[1]
        num_anchors = arg.shape[2]
        # repeat matrix num_anchors times
        A_ = A[:, index:index+num_parts].unsqueeze(2).repeat(1, 1, num_anchors, 1, 1)
        tr = torch.matmul(A_, arg)
        transformed.append(tr)
        index += num_parts

    return transformed

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def transform_template(input, params):
    size = input.shape[2]
    # scale up translation
    params[..., -1] = params[..., -1] * size
    return kornia.geometry.warp_affine(input, params, dsize=(size, size))

def load_anchor_points(path, batch_size=1):
    """
    load anchor points from json file
    change this according to your definitions
    Args:
        anchor_points: json file containing anchor points per part in column, row format similar to open-cv
        device: torch.device, either cpu or gpu
    """
    with open(path, 'r') as file: anchor_points = json.load(file)
    # assumes three anchor points for core, two (parent+child) for all others except hands and feet and head
    # change this accordingly for different template definitions!
    double = []
    single = []
    for k, v in anchor_points.items():
        if k in ['left hand', 'right hand', 'left foot', 'right foot', 'head']:
            single.append(v)
        elif k == 'core':
            triple = [v]
        else:
            double.append(v)

    return torch.tensor(triple).float().unsqueeze(-1).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1), \
           torch.tensor(single).float().unsqueeze(-1).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1), \
           torch.tensor(double).float().unsqueeze(-1).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

def compute_anchor_loss(core, double, single, size):
    """
    compute mean distance between pairs of transformed anchor_points,
    change this according to your connectivity constraints
    """
    loss = 0
    # normalize to range 0, 1
    core = core/size
    single = single/size
    double = double/size

    # loss between core and hips and shoulders
    indices1 = [0, 0, 1, 1]
    # hips and shoulders
    indices2 = [0, 1, 6, 7]

    for index1, index2 in zip(indices1, indices2):

        loss += nn.MSELoss()(core[:, 0, index1], double[:, index2, 0])
    # head and core
    loss += nn.MSELoss()(core[:, 0, -1], single[:, -1, 0])

    # hips to thighs to shins, shoulders to arms to forearms
    indices3 = [0, 1, 2, 3, 6, 7, 8, 9]
    indices4 = [2, 3, 4, 5, 8, 9, 10, 11]

    for index3, index4 in zip(indices3, indices4):
        loss += nn.MSELoss()(double[:, index3, 1], double[:, index4, 0])

    #  shin to feet, forarms to hands
    indices5 = [4, 5, 10, 11]
    indices6 = [0, 1, 2, 3]

    for index5, index6 in zip(indices5, indices6):

        loss += nn.MSELoss()(double[:, index5, 1], single[:, index6, 0])

    return loss

def compute_boundary_loss(core, single, double, img_size):
    """
    compute boundary loss, boundaries are 0 and 1
    loss = x if x smaller or greater than 0, 1
    0 otherwise
    """
    core = core.view(core.shape[0], -1, core.shape[3])
    single = core.view(single.shape[0], -1, single.shape[3])
    double = core.view(double.shape[0], -1, double.shape[3])

    comb = torch.cat([core, single, double], dim=1)

    # normalize to range -1  to 1
    comb = (comb / img_size) * 2 - 1

    return nn.Threshold(1, 0)(torch.abs(comb)).sum(1).mean()

def draw_shape(pos, sigma_x, sigma_y, angle, size):
    """
    draw (batched) gaussian with sigma_x, sigma_y on 2d grid
    Args:
        pos: torch.tensor (float) with shape (2) specifying center of gaussian blob (x: row, y:column)
        sigma_x: torch.tensor (float scalar), scaling parameter along x-axis
        sigma_y: similar along y-axis
        angle: torch.tensor (float scalar) rotation angle in radians
        size: int specifying size of image
        device: torch.device, either cpu or gpu
    Returns:
        torch.tensor (1, 1, size, size) with gaussian blob
    """
    device = pos.device
    assert sigma_x.device == sigma_y.device == angle.device == device, "inputs should be on the same device!"

    # create 2d meshgrid
    x, y = torch.meshgrid(torch.arange(0, size), torch.arange(0, size))
    x, y = x.unsqueeze(0).unsqueeze(0).to(device), y.unsqueeze(0).unsqueeze(0).to(device)

    # see https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
    a = torch.cos(angle) ** 2 / (2 * sigma_x ** 2) + torch.sin(angle) ** 2 / (2 * sigma_y ** 2)
    b = -torch.sin(2 * angle) / (4 * sigma_x ** 2) + torch.sin(2 * angle) / (4 * sigma_y ** 2)
    c = torch.sin(angle) ** 2 / (2 * sigma_x ** 2) + torch.cos(angle) ** 2 / (2 * sigma_y ** 2)

    # append dimsensions for broadcasting
    pos = pos.view(1, 1, 2, 1, 1)
    a, b, c = a.view(1, 1), b.view(1, 1), c.view(1, 1)

    # pixel-wise distance from center
    xdist = (x - pos[:, :, 0])
    ydist = (y - pos[:, :, 1])

    # gaussian function
    g = torch.exp((-a * xdist ** 2 - 2 * b * xdist * ydist - c * ydist ** 2))

    return g

def draw_template(path, size, batch_size=1):
    """
    draw template consisting of limbs defined by gaussian heatmap
    Args:
        template: json file defining all parts
        size: int, image size (assumed quadratic), this should match the center coordinates defined in the json!
        device: torch.device, either cpu or gpu
    """
    with open(path, 'r') as file:
        template = json.load(file)
    heatmaps = []
    for v in template.values():
        center = torch.tensor(v['center'])
        sx = torch.tensor(v['sx'])
        sy = torch.tensor(v['sy'])
        angle = torch.tensor(v['angle'])
        heatmaps.append(draw_shape(center, sx, sy, angle, size))

    heatmaps = torch.cat(heatmaps, dim=1).repeat(batch_size, 1, 1, 1)

    return heatmaps

class ShapeTemplatesLitModule (BaseLitModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda1 = self.config['loss_args']['anchor_loss_weight']
        self.lambda2 = self.config['loss_args']['boundary_loss_weight']
        anchor_pts_path = self.config['exp_args']['anchor_points_path']

        ################# Need to modify #########################
        template = draw_template(self.config['exp_args']['template_path'], size=256)
        self.register_buffer('template', template)
        ###############################################
        core, single, double = load_anchor_points(anchor_pts_path)
        self.register_buffer('core', core)
        self.register_buffer('single', single)
        self.register_buffer('_double', double)

        I = torch.eye(3)[0:2].view(1, 1, 2, 3).repeat(1, self.config['model_args']['num_parts'], 1, 1)
        self.register_buffer('I', I)
    
    def training_step(self, batch, batch_idx):
        frame1 = batch['a']['I']
        frame2 = batch['b']['I']

        loss, transformed_template, reconstructed = self._training_step_call(frame1, frame2)

        out = {
            'loss': loss
        }
        self.write_logger(out, frame1, transformed_template, reconstructed)
        return out
        
    
    def _training_step_call(self, frame1, frame2):
        template = self.template.repeat(frame1.shape[0], 1, 1, 1)
        I = self.I.repeat(frame1.shape[0], 1, 1, 1)
        bs, c, img_size, img_size = frame1.shape
        estimated_params = self.model.regressor(frame1)
        estimated_params = I + estimated_params
        num_parts = estimated_params.shape[1]

        batched_template = template.flatten(0, 1).unsqueeze(1)
        batched_params = estimated_params.view(-1, 2, 3)
        transformed_template = transform_template(batched_template, batched_params).view(bs, -1, img_size, img_size)
        
        A = torch.tensor([0, 0, 1], device=self.device).view(1, 1, 1, 3).repeat(bs, num_parts, 1, 1)
        A = torch.cat([estimated_params, A], dim=-2)
        transformed_anchors = transform_anchor_points(A, self.core.repeat(bs,1,1,1,1), self._double.repeat(bs,1,1,1,1), self.single.repeat(bs,1,1,1,1))

        reconstructed = self.model.translator(frame2, transformed_template)
        anchor_loss = compute_anchor_loss(*transformed_anchors, size=img_size)
        boundary_loss = compute_boundary_loss(*transformed_anchors, img_size=img_size)
        recon_loss = nn.L1Loss()(self.model.vgg(frame1.repeat(1, 3, 1, 1)), self.model.vgg(reconstructed.repeat(1, 3, 1, 1)))
        loss = recon_loss + self.lambda1 * anchor_loss + self.lambda2 * boundary_loss

        return loss, transformed_template, reconstructed
    
    def DummyForward(self):
        obj = self.train_dataloader()
        for one in obj: break
        frame1 = one['a']['I']
        frame2 = one['b']['I']
        self._training_step_call(frame1, frame2)
    
    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        pass

    def validation_epoch_end(self, val_outs):
        print ('Validation finish')
    
    def write_logger(self, loss_dict, frame1, transformed_template, reconstructed):
        for key, val in loss_dict.items(): self.log('Loss/%s'%key, val, on_step=True)
        if self.global_step % self.config['exp_args']['exp_freq'] == 0:
            transformed_template = torch.sum(transformed_template, dim=1, keepdim=True).clamp(0, 1)
            self.logger.experiment.add_images('Image/I1', frame1, self.global_step)
            self.logger.experiment.add_images('Image/Transformed', transformed_template, self.global_step)
            self.logger.experiment.add_images('Image/Reconstructed', reconstructed, self.global_step)
