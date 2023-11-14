import torch
import torch.nn as nn
import numpy as np
from .BaseTrainer import BaseLitModule


class DHP19LitModule (BaseLitModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        img = batch['img']
        y_heatmaps = batch['y_heatmaps']
        pred = self.model(img)

        #loss = nn.MSELoss()(pred, y_heatmaps)
        a = torch.mean((pred - y_heatmaps)**2, dim=1)
        b = torch.mean(a, dim=-1)
        c = torch.mean(b, dim=-1)
        loss = torch.sum(c)
        out = {
            'loss': loss,
            'MSELoss': loss
        }
        self.write_logger(out)

        return out
    
    def write_logger(self, loss_dict):
        for key, val in loss_dict.items(): self.log('Loss/%s'%key, val, on_step=True)

    def on_train_epoch_end(self):
        if self.global_rank == 0:
            if self.val_results is not None:
                print (self.val_results)
                self.WriteValResults(self.val_results)
                self.model.Save(self.current_epoch, accuracy=-self.val_results['mpjpe'], replace=True)
            else:
                print ('No val results!')
            self.val_results = None

    
    def on_validation_epoch_start(self):
        if self.global_rank == 0:
            self.val_bag = {
                'pred_vu': torch.zeros(len(self.val_data), 13, 2).long(),
                'GT_vu': torch.zeros(len(self.val_data), 13, 2).long(),
                'GT_mask': torch.zeros(len(self.val_data), 13).bool(),
            }

    def validation_step(self, batch, batch_idx):
        img = batch['img']
        #y_heatmaps = batch['y_heatmaps']
        #y_2d = batch['y_2d']
        #gt_mask = batch['gt_mask']
        pred = self.model(img)

        out = {
            'pred': pred.cpu()
        }

        return out
        

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        idx = self.all_gather(batch['idx']).flatten(0, 1) # bs
        pred = self.all_gather(outputs['pred']).flatten(0, 1) # bs x 13 x 260 x 344
        y_2d = self.all_gather(batch['y_2d']).flatten(0, 1) # bs x 13 x 2
        gt_mask = self.all_gather(batch['gt_mask']).flatten(0, 1) # bs x 13

        h = pred.shape[2]
        w = pred.shape[3]
        if self.global_rank == 0:
            idx = idx.cpu()
            pred = pred.cpu()
            y_2d = y_2d.cpu()
            gt_mask = gt_mask.cpu()

            mx_indices = torch.max(pred.flatten(-2, -1), dim=-1, keepdim=True)[1]
            mx_v = torch.trunc(mx_indices / w)
            mx_u = torch.remainder(mx_indices, w)
            mx_coord = torch.cat([mx_v, mx_u], dim=-1) # bs x 13 x 2
            self.val_bag['pred_vu'][idx, ...] = mx_coord.long()
            self.val_bag['GT_vu'][idx, ...] = y_2d.long()
            self.val_bag['GT_mask'][idx, ...] = gt_mask.bool()
        
        del outputs['pred']

    def validation_epoch_end(self, val_outs):
        if self.global_rank == 0:
            pred_vu = self.val_bag['pred_vu']
            GT_vu = self.val_bag['GT_vu']
            GT_mask = self.val_bag['GT_mask']
            self.val_results = self.val_data.evaluate(pred_vu.numpy(), GT_vu.numpy(), GT_mask.numpy())