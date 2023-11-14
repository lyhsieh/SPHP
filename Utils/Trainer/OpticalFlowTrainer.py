import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from imageio import imwrite
from .BaseTrainer import BaseLitModule


def charbonnier_loss(delta, alpha=0.45, epsilon=1e-3):
    loss = torch.mean(torch.pow(torch.mul(delta,delta) + torch.mul(epsilon,epsilon), alpha))
    return loss

def compute_smoothness_loss(flows):
    total_smoothness_loss = 0.
    loss_weight_sum = 0.

    for flow in flows:
        # [B, C, H, W]
        flow_ucrop = flow[:, :, 1:, :]
        flow_dcrop = flow[:, :, :-1, :]
        flow_lcrop = flow[:, :, :, 1:]
        flow_rcrop = flow[:, :, :, :-1]

        flow_ulcrop = flow[:, :, 1:, 1:]
        flow_drcrop = flow[:, :, :-1, :-1]
        flow_dlcrop = flow[:, :, :-1, 1:]
        flow_urcrop = flow[:, :, 1:, :-1]

        smoothness_loss = charbonnier_loss(flow_lcrop - flow_rcrop) + \
                          charbonnier_loss(flow_ucrop - flow_dcrop) + \
                          charbonnier_loss(flow_ulcrop - flow_drcrop) + \
                          charbonnier_loss(flow_dlcrop - flow_urcrop)
        total_smoothness_loss += smoothness_loss / 4.
        loss_weight_sum += 1.
    total_smoothness_loss /= loss_weight_sum
    return total_smoothness_loss

def compute_photometric_loss(multi_scale_flows, img1, img2, offset):
    total_photometric_loss = 0.
    loss_weight_sum = 0.

    for flow in multi_scale_flows:
        B, C, H, W = flow.size()
        pre_img = F.interpolate(img1, (H, W), mode='bilinear')
        cur_img = F.interpolate(img2, (H, W), mode='bilinear')
        offset_down = F.interpolate(offset, (H, W), mode='bilinear')

        coord = (offset_down + flow).permute(0, 2, 3, 1)
        warped_pre_img = F.grid_sample(cur_img, coord, align_corners=True)

        photometric_loss = charbonnier_loss(pre_img - warped_pre_img)
        total_photometric_loss += photometric_loss
        loss_weight_sum += 1.
    total_photometric_loss /= loss_weight_sum
    return total_photometric_loss, warped_pre_img

def compute_photometric_loss_residual(multi_scale_flows, img1, img2, UV2, offset):
    total_photometric_loss = 0.
    loss_weight_sum = 0.

    for flow_residual in multi_scale_flows:
        B, C, H, W = flow_residual.size()
        pre_img = F.interpolate(img1, (H, W), mode='bilinear')
        cur_img = F.interpolate(img2, (H, W), mode='bilinear')
        offset_down = F.interpolate(offset, (H, W), mode='bilinear')
        UV2_down = F.interpolate(UV2, (H, W), mode='bilinear')
        flow = UV2_down + flow_residual
        coord = (offset_down + flow).permute(0, 2, 3, 1)
        warped_pre_img = F.grid_sample(cur_img, coord, align_corners=True)

        photometric_loss = charbonnier_loss(pre_img - warped_pre_img)
        total_photometric_loss += photometric_loss
        loss_weight_sum += 1.
    total_photometric_loss /= loss_weight_sum
    return total_photometric_loss, warped_pre_img


class OpticalFlowLitModule(BaseLitModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        I1 = batch['a']['I']
        UV1 = batch['a']['UV']
        I2 = batch['b']['I']
        UV2 = batch['b']['UV']
        offset = batch['offset']

        UV_cat = torch.cat([UV1, UV2], dim=1)
        flow_pyramid = self.model(UV_cat)

        photometric_loss, warped_img = compute_photometric_loss(flow_pyramid, I1, I2, offset)
        smoothness_loss = compute_smoothness_loss(flow_pyramid)
        loss = photometric_loss + 0.5 * smoothness_loss

        out = {
            'loss': loss,
            'photometric-loss': photometric_loss,
            'smoothness-loss': smoothness_loss
        }
        self.write_logger(out, I1, I2, UV1, UV2, warped_img, flow_pyramid[-1])
        return out
    
    def on_train_epoch_end(self):
        param = self.model.state_dict()
        torch.save(param, 'save/%.5d.pkl'%self.current_epoch)

    def validation_step(self, batch, batch_idx):
        idx = batch['idx']
        I1 = batch['a']['I']
        UV1 = batch['a']['UV']
        I2 = batch['b']['I']
        UV2 = batch['b']['UV']
        offset = batch['offset']

        UV_cat = torch.cat([UV1, UV2], dim=1)
        flow_pyramid = self.model(UV_cat)
        _, warped_img = compute_photometric_loss(flow_pyramid, I1, I2, offset)

        if True:
            '''
            img = torch.cat([I1, I2], dim=-1)[:, 0, ...]
            #uv = (torch.cat([UV1, UV2], dim=-1) + 1) / 2
            uv = (UV2 + 1) / 2
            u = uv[:, 0, ...]
            v = uv[:, 1, ...]
            pred_uv = (flow_pyramid[-1] + 1) / 2
            pred_u = pred_uv[:, 0, ...]
            pred_v = pred_uv[:, 1, ...]
            
            u_map = torch.cat([u, pred_u], dim=-1)
            v_map = torch.cat([u, pred_v], dim=-1)

            big = torch.cat([img, u_map, v_map], dim=1)[:, None, ...]
            big = F.interpolate(big, scale_factor=1.0/3, mode='bilinear', align_corners=True)

            idx = self.all_gather(idx).flatten(0, 1)
            big = self.all_gather(big).flatten(0, 1)[:, 0, ...]
            '''
            img = torch.cat([I1, I2, warped_img], dim=-1)
            idx = self.all_gather(idx).flatten(0, 1)
            big = self.all_gather(img).flatten(0, 1)[:, 0, ...]

            for i in range(idx.shape[0]):
                path = '%s/%.5d.jpg'%(self.config['exp_args']['val_results_path'], idx[i])
                fig = (big[i].cpu().numpy() * 255).astype(np.uint8)
                imwrite(path, fig)
                
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        pass

    def validation_epoch_end(self, val_outs):
        print ('Validation finish')
    
    def write_logger(self, loss_dict, I1, I2, UV1, UV2, warped_img, flow):
        if self.global_rank == 0:
            for key, val in loss_dict.items(): self.log('Loss/%s'%key, val, on_step=True)
            if self.global_step % self.config['exp_args']['exp_freq'] == 0:
                self.logger.experiment.add_images('Image/I1', I1, self.global_step)
                self.logger.experiment.add_images('Image/I2', I2, self.global_step)
                self.logger.experiment.add_images('Image/Warp', warped_img, self.global_step)
                self.logger.experiment.add_images('Image/UV2U', (UV2[:, 0:1, ...]+1)/2, self.global_step)
                self.logger.experiment.add_images('Image/UV2V', (UV2[:, 1:, ...]+1)/2, self.global_step)
                self.logger.experiment.add_images('Image/FlowU', (flow[:, 0:1, ...]+1)/2, self.global_step)
                self.logger.experiment.add_images('Image/FlowV', (flow[:, 1:, ...]+1)/2, self.global_step)