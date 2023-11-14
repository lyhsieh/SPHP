'''
keypoint id:
 0: head      1: shoulderR  2: shoulderL  3: elbowR   4: elbowL
 5: hipR      6: hipL       7: handR      8: handL    9: kneeR
10: kneeL    11: footR     12: footL
'''
import glob
import cv2
import numpy as np
from imageio import imread
from ..BaseDataset import BaseDataset



def decay_heatmap(heatmap, sigma2=4):
    '''
    Perform Gaussian Blur to calculate the heatmaps
    '''
    heatmap = cv2.GaussianBlur(heatmap, (0,0), sigma2)
    heatmap /= np.max(heatmap) # keep the max to 1
    return heatmap

def generateGT(vicon_xyz, P_mat_cam, sigma):
    '''
    generate ground truth heatmap
    '''
    H, W = 288, 384
    num_joints = 13
    vicon_xyz_homog = np.concatenate([vicon_xyz, np.ones([1, 13])], axis=0)
    coord_pix_homog = np.matmul(P_mat_cam, vicon_xyz_homog)
    coord_pix_homog_norm = coord_pix_homog / coord_pix_homog[-1]
    u = coord_pix_homog_norm[0] # x
    v = H - coord_pix_homog_norm[1] # y

    mask = np.ones(u.shape).astype(np.float32)
    mask[np.isnan(u)] = 0
    mask[np.isnan(v)] = 0

    mask[u>W] = 0
    mask[u<=0] = 0
    mask[v>H] = 0
    mask[v<=0] = 0

    u = u.astype(np.int32)
    v = v.astype(np.int32)
    label_heatmaps = np.zeros((H, W, num_joints))
    for fmidx, zipd in enumerate(zip(v, u, mask)):
        if zipd[2]==1: # write joint position only when projection within frame boundaries
            label_heatmaps[zipd[0], zipd[1], fmidx] = 1
            label_heatmaps[:,:,fmidx] = decay_heatmap(label_heatmaps[:,:,fmidx], sigma2=sigma)
    return np.stack((v,u), axis=-1), mask, label_heatmaps

def get_heatmap(label, sigma):
    '''
    heatmap calculation
    '''
    H = 288
    W = 384
    num_joints = 13
    mask = np.ones(13).astype(np.float32)
    label_heatmaps = np.zeros((H, W, num_joints))

    for i in range(num_joints):
        label_heatmaps[label[i,0], label[i,1], i] = 1
        label_heatmaps[:,:,i] = decay_heatmap(label_heatmaps[:,:,i], sigma2=sigma)

    return mask, label_heatmaps


def ComputeCameraParameters(calib_path):
    '''
    Compute camera parameters
    '''
    src_shape = [480, 640]
    h, w = src_shape
    data = np.load(calib_path, allow_pickle=True).item()
    K, dist = data['K'], data['dist']
    K_optim, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 0, (w,h))
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, K_optim, src_shape[::-1], 5)
    return mapx, mapy


class SPHPDataset (BaseDataset):
    '''
    define SPHP Dataset
    '''
    def __init__(self, dataset_path, aug, subject, camera, movement_class,
                 img_format, calib_path, sigma=4, **kwargs):
        super().__init__(**kwargs)
        self.aug = aug
        self.subject = subject
        self.camera = camera
        self.sigma = sigma
        self.img_format = img_format
        self.mapx, self.mapy = ComputeCameraParameters(calib_path)
        self.label_list = []
        self.edge_list = []
        self.gray_list = []
        self.mvv_list = []
        self.mvh_list = []

        assert img_format in ['EDG','GRA','FUS','MV']
        assert movement_class in ['C1', 'C2', 'C3', 'C4', 'ALL']
        movement_dic = {
            'C1': ['01', '02', '03', '04'],
            'C2': ['05', '06', '07'],
            'C3': ['09', '10', '14', '16'],
            'C4': ['08', '11', '12', '13', '15'],
            'ALL': ['01', '02', '03', '04', '05', '06', '07', '08',\
                    '09', '10', '11', '12', '13', '14', '15', '16']
        }

        for cam in camera:
            for sub in subject:
                for move in movement_dic[movement_class]:
                    label_path =  f'{dataset_path}/{cam}/{sub}/{move}/pose_change'
                    self.label_list += sorted(glob.glob(f'{label_path}/*'))
                    edg_path = f'{dataset_path}/{cam}/{sub}/{move}/EDG'
                    self.edge_list += sorted(glob.glob(f'{edg_path}/*'))
                    gra_path = f'{dataset_path}/{cam}/{sub}/{move}/GRA'
                    self.gray_list += sorted(glob.glob(f'{gra_path}/*'))
                    mvv_path = f'{dataset_path}/{cam}/{sub}/{move}/MVV'
                    self.mvv_list += sorted(glob.glob(f'{mvv_path}/*'))
                    mvh_path = f'{dataset_path}/{cam}/{sub}/{move}/MVH'
                    self.mvh_list += sorted(glob.glob(f'{mvh_path}/*'))

        print(len(self.label_list))
        assert len(self.label_list) == len(camera) * len(subject)* \
                                       len(movement_dic[movement_class]) * 300
        assert len(self.edge_list) == len(self.gray_list) == len(self.mvh_list)\
               == len(self.mvv_list) == len(self.label_list)

    def __len__(self,):
        return len(self.label_list)

    def __getitem__(self, idx):
        if self.img_format in ['EDG', 'GRA']:
            if self.img_format == 'EDG':
                name = self.edge_list[idx]
            elif self.img_format == 'GRA':
                name = self.gray_list[idx]
            img = np.asarray(imread(name))
            img = cv2.remap(img, self.mapx, self.mapy, cv2.INTER_CUBIC)
            img = cv2.resize(img, (384, 288), interpolation=cv2.INTER_AREA)
            img = (img.astype(np.float32)/ 255.0)[None, ...]

        elif self.img_format in ['FUS', 'MV']:
            img_list = []
            for name in [self.mvv_list[idx],self.mvh_list[idx]]:
                img = np.asarray(imread(name))
                img = cv2.remap(img, self.mapx, self.mapy, cv2.INTER_CUBIC)
                img = cv2.resize(img, (384, 288), interpolation=cv2.INTER_AREA)
                img = img.astype(np.float32) - 128.0
                img = img / 128.0
                img_list.append(img)

            if  self.img_format == 'MV':
                img =  np.stack((img_list[0],img_list[1]))
            else:
                name = self.edge_list[idx]
                img = np.asarray(imread(name))
                img = cv2.remap(img, self.mapx, self.mapy, cv2.INTER_CUBIC)
                img = cv2.resize(img, (384, 288), interpolation=cv2.INTER_AREA)
                img = img.astype(np.float32) / 255.0
                img_list.append(img)
                img =  np.stack((img_list[2],img_list[0],img_list[1]))

        y_2d = np.load(self.label_list[idx]) # x, y
        for j in range(y_2d.shape[0]):
            tmp = y_2d[j,0]
            y_2d[j, 0] = int((float(y_2d[j,1]) / 480.0) * 288.0)
            y_2d[j, 1] = int((float(tmp) / 640.0) * 384.0)
        gt_mask, y_heatmaps = get_heatmap(y_2d, self.sigma)
        y_heatmaps = y_heatmaps.transpose(2, 0, 1).astype(np.float32)

        if self.aug:
            if np.random.rand() > 0.5:
                img = img[:, :, ::-1].copy()
                y_heatmaps = y_heatmaps[:, :, ::-1].copy()
                idx_lst = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]
                lst = []
                for i in idx_lst:
                    lst.append(y_heatmaps[i:i+1, ...])
                y_heatmaps = np.concatenate(lst, axis=0)

        out = {
            'idx': idx,
            'img': img,
            'y_2d': y_2d, # y, x
            'gt_mask': gt_mask,
            'y_heatmaps': y_heatmaps
        }
        return out

    def evaluate(self, pred_vu, GT_vu, GT_mask):
        '''
        calculate MPJPE
        '''
        # GT_mask is # x 13
        assert len(GT_mask.shape) == 2
        assert pred_vu.shape == GT_vu.shape # total # of img x 13 x 2

        GT_mask = np.tile(GT_mask[..., None], (1, 1, 2)).astype(bool)
        GT_vu = GT_vu.astype(np.float32)
        GT_vu[~GT_mask] = np.nan

        dist_2d = np.linalg.norm((GT_vu - pred_vu), axis=-1)
        mpjpe = np.nanmean(dist_2d, axis=-1)
        avg_mpjpe = np.nanmean(mpjpe, axis=-1)

        out = {
            'mpjpe': avg_mpjpe
        }
        return out
