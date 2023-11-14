import os
from sklearn.covariance import EmpiricalCovariance
import torch
from .BaseTrainer import BaseLitModule
from .. import Evaluations, PostProcessing


class BottomUpLitModule(BaseLitModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser = PostProcessing.Group.HeatmapParser(self.config['testing_args'])

    def training_step(self, batch, batch_idx):
        pred = self.model(batch['img'])
        losses = self.model.calculate_loss(pred, batch['targets'], batch['masks'], batch['joints'])

        heatmap_loss = losses['heatmap_loss']
        push_loss = losses['push_loss']
        pull_loss = losses['pull_loss']

        total_loss = heatmap_loss + push_loss + pull_loss        
        losses['total_loss'] = total_loss

        out = {
            'loss': total_loss
        }

        self.write_logger(losses)

        return out
    
    def write_logger(self, loss_dict):
        for key, val in loss_dict.items(): self.log('Loss/%s'%key, val, on_step=True)
    
    def validation_step(self, batch, batch_idx):
        ann_info = batch['ann_info']
        test_scale_factor = ann_info['test_scale_factor'][0]
        base_size = ann_info['base_size']
        center = ann_info['center'][0].cpu().numpy()
        scale = ann_info['scale'][0].cpu().numpy()
        test_cfg = self.config['testing_args']


        aug_data = ann_info['aug_data']
        scale_heatmaps_list = []
        scale_tags_list = []
        result = {}
        for idx, s in enumerate(sorted(test_scale_factor, reverse=True)):
            image_resized = aug_data[idx]
            assert image_resized.shape[0] == 1 and image_resized.shape[1] == 1
            image_resized = image_resized.squeeze(0)
            outputs = self.model(image_resized)

            heatmaps, tags = Evaluations.BottomUpEvaluation.split_ae_outputs(
                outputs, 
                test_cfg['num_joints'],
                test_cfg['with_heatmaps'], 
                test_cfg['with_ae'],
                test_cfg.get('select_output_index', range(len(outputs)))
            )
            if test_cfg.get('flip_test', True):
                outputs_flipped = self.model(torch.flip(image_resized, [3]))
                heatmaps_flipped, tags_flipped = Evaluations.BottomUpEvaluation.split_ae_outputs(
                    outputs_flipped, 
                    test_cfg['num_joints'],
                    test_cfg['with_heatmaps'], 
                    test_cfg['with_ae'],
                    test_cfg.get('select_output_index', range(len(outputs)))
                )
                heatmaps_flipped = Evaluations.BottomUpEvaluation.flip_feature_maps(
                    heatmaps_flipped, 
                    flip_index=ann_info['flip_index']
                )
                if test_cfg['tag_per_joint']:
                    tags_flipped = Evaluations.BottomUpEvaluation.flip_feature_maps(
                        tags_flipped, 
                        flip_index=ann_info['flip_index']
                    )
                else:
                    tags_flipped = Evaluations.BottomUpEvaluation.flip_feature_maps(
                        tags_flipped, 
                        flip_index=None, 
                        flip_output=True
                    )
            else:
                heatmaps_flipped = None
                tags_flipped = None
            
            aggregated_heatmaps = Evaluations.BottomUpEvaluation.aggregate_stage_flip(
                heatmaps,
                heatmaps_flipped,
                index=-1,
                project2image=test_cfg['project2image'],
                size_projected=base_size,
                align_corners=test_cfg.get('align_corners', True),
                aggregate_stage='average',
                aggregate_flip='average')

            aggregated_tags = Evaluations.BottomUpEvaluation.aggregate_stage_flip(
                tags,
                tags_flipped,
                index=-1,
                project2image=test_cfg['project2image'],
                size_projected=base_size,
                align_corners=test_cfg.get('align_corners', True),
                aggregate_stage='concat',
                aggregate_flip='concat')
            
            if s == 1 or len(test_scale_factor) == 1:
                if isinstance(aggregated_tags, list):
                    scale_tags_list.extend(aggregated_tags)
                else:
                    scale_tags_list.append(aggregated_tags)

            if isinstance(aggregated_heatmaps, list):
                scale_heatmaps_list.extend(aggregated_heatmaps)
            else:
                scale_heatmaps_list.append(aggregated_heatmaps)
        aggregated_heatmaps = Evaluations.BottomUpEvaluation.aggregate_scale(
            scale_heatmaps_list,
            align_corners=test_cfg.get('align_corners', True),
            aggregate_scale='average')

        aggregated_tags = Evaluations.BottomUpEvaluation.aggregate_scale(
            scale_tags_list,
            align_corners=test_cfg.get('align_corners', True),
            aggregate_scale='unsqueeze_concat')
        heatmap_size = aggregated_heatmaps.shape[2:4]
        tag_size = aggregated_tags.shape[2:4]
        if heatmap_size != tag_size:
            tmp = []
            for idx in range(aggregated_tags.shape[-1]):
                tmp.append(
                    torch.nn.functional.interpolate(
                        aggregated_tags[..., idx],
                        size=heatmap_size,
                        mode='bilinear',
                        align_corners=test_cfg.get('align_corners',
                                                        True)).unsqueeze(-1))
            aggregated_tags = torch.cat(tmp, dim=-1)
        
        grouped, scores = self.parser.parse(
            aggregated_heatmaps,
            aggregated_tags,
            test_cfg['adjust'],
            test_cfg['refine']
        )
        
        preds = Evaluations.BottomUpEvaluation.get_group_preds(
            grouped,
            center,
            scale, 
            [aggregated_heatmaps.size(3), aggregated_heatmaps.size(2)],
            use_udp=test_cfg.get('use_udp', False)
        )
        
        image_paths = []
        image_paths.append(batch['image_file'][0])
        
        if True:
            output_heatmap = aggregated_heatmaps.detach().cpu().numpy()
        else:
            output_heatmap = None

        result['preds'] = preds
        result['scores'] = scores
        result['image_paths'] = image_paths
        #result['output_heatmap'] = output_heatmap

        return result
    
    def validation_epoch_end(self, val_outs):
        if self.global_rank == 0:
            p = self.config['exp_args']['val_results_path']
            lst = ['%s/%s'%(p, x) for x in sorted(os.listdir(p)) if x.endswith('.pkl')]

            results = []
            for one in lst:
                tmp = torch.load(one)
                results.append(tmp)
            
            self.val_results = self.val_data.evaluate(results)
