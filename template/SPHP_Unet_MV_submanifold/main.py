'''
Training script for SPHP on Unet backbone
'''
import sys
import argparse
import yaml
import torch
sys.path.append('../..')
import Utils

from pytorch_lightning.loggers import TensorBoardLogger
from network import MyModel


class MM(Utils.Trainer.DHP19LitModule):
    '''
    my model
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        img = batch['img']
        y_heatmaps = batch['y_heatmaps']
        pred = self.model(img)

        temp1 = torch.mean((pred - y_heatmaps) ** 2, dim=1)
        temp2 = torch.mean(temp1, dim=-1)
        temp3 = torch.mean(temp2, dim=-1)
        loss = torch.sum(temp3)

        loss_log = {
            'MSELoss': loss
        }
        if self.logger is not None:
            for key, val in loss_log.items():
                self.log(f'Loss/{key}', val, on_step=True)

        out = {
            'loss': loss
        }
        return out


def main(args, config):
    '''
    main function
    '''
    Utils.Tools.fixSeed(config['exp_args']['seed'])
    model = MyModel(**config['model_args'])
    model.Load(args.epoch)
    litmodule = MM(config, model)

    if args.mode == 'train':
        logger = TensorBoardLogger(config['exp_args']['exp_path'])
        strategy = 'ddp_find_unused_parameters_false'
        trainer = Utils.Trainer.BaseTrainer(
            accelerator='gpu',
            strategy=strategy,
            enable_progress_bar=False,
            max_epochs=config['exp_args']['epoch'],
            num_sanity_val_steps=0,
            logger=logger,
            enable_checkpointing=False,
            num_nodes=1,
            check_val_every_n_epoch=1,
            devices=args.device,
        )
        trainer.fit(model=litmodule)

    elif args.mode == 'val':
        strategy = 'ddp'
        trainer = Utils.Trainer.BaseTrainer(
            accelerator='gpu',
            strategy=strategy,
            enable_progress_bar=False,
            max_epochs=config['exp_args']['epoch'],
            num_sanity_val_steps=0,
            logger=False
        )
        trainer.validate(model=litmodule)
        if litmodule.global_rank == 0:
            print(litmodule.val_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for SPHP on DHP19',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode',
                        default='train',
                        type=str,
                        choices=['train', 'val'],
                        help='train/val mode')
    parser.add_argument('--epoch', type=int, help='load epoch')
    parser.add_argument('--device', default=1, type=int, help='# of available GPUs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')

    args = parser.parse_args()

    with open('./config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['dataset_args']['train']['loader_args']['batch_size'] = args.batch_size
    config['dataset_args']['val']['loader_args']['batch_size'] = args.batch_size

    main(args, config)
