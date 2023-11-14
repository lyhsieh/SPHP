import os
import abc
import numpy as np
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning import loggers as pl_loggers
from .. import Tools, Datasets


def ScriptStart(args, config, litmodule):
    if args.mode == 'train':
        logger_type = config['exp_args'].get('logger_type', 'TensorBoardLogger')
        num_nodes = config['exp_args'].get('num_nodes', 1)
        check_val_every_n_epoch = config['exp_args'].get('check_val_every_n_epoch', 1)
        limit_val_batches = config['exp_args'].get('limit_val_batches', 1.0)
        devices = config['exp_args'].get('devices', None)
        if logger_type == 'TensorBoardLogger':
            logger = pl_loggers.TensorBoardLogger(config['exp_args']['exp_path'])
        else:
            raise ValueError('Logger type weird')
        trainer = BaseTrainer(
            accelerator='gpu',
            strategy=DDPStrategy(find_unused_parameters=False),
            enable_progress_bar=False,
            max_epochs=config['exp_args']['epoch'],
            num_sanity_val_steps=0,
            logger=logger,
            enable_checkpointing=False,
            check_val_every_n_epoch=check_val_every_n_epoch,
            limit_val_batches=limit_val_batches,
            num_nodes=num_nodes,
            devices=devices
        )
        trainer.fit(model=litmodule)
    else:
        num_nodes = config['exp_args'].get('num_nodes', 1)
        devices = config['exp_args'].get('devices', None)
        trainer = BaseTrainer(
            accelerator='gpu',
            strategy=DDPStrategy(find_unused_parameters=False),
            enable_progress_bar=False,
            max_epochs=config['exp_args']['epoch'],
            num_sanity_val_steps=0,
            logger=False,
            num_nodes=num_nodes,
            devices=devices
        )
        trainer.validate(model=litmodule)
        if litmodule.global_rank == 0:
            print (litmodule.val_results)


class MyProgressCallback(pl.callbacks.Callback):
    def __init__(self, tqdm_total, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tqdm_total = tqdm_total

    def on_train_epoch_start(self, trainer, pl_module):
        if pl_module.global_rank == 0:
            count = int(np.ceil(len(pl_module.train_dataloader_obj) / self.tqdm_total))
            self.myprogress = Tools.MyTqdm(range(count))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        if pl_module.global_rank == 0: next(self.myprogress)

    def on_validation_start(self, trainer, pl_module):
        if pl_module.global_rank == 0:
            tmp = len(pl_module.val_dataloader_obj)
            count = int(np.ceil(tmp / self.tqdm_total))
            self.myprogress = Tools.MyTqdm(range(count))

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if pl_module.global_rank == 0: next(self.myprogress)

class BaseTrainer(pl.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.strategy, pl.strategies.dp.DataParallelStrategy):
            tqdm_total = 1
        elif isinstance(self.strategy, pl.strategies.ddp.DDPStrategy):
            tqdm_total = self.num_nodes * self.num_devices
        else:
            raise NotImplementedError
        self.callbacks.append(MyProgressCallback(tqdm_total=tqdm_total))

        from pytorch_lightning.callbacks import LearningRateMonitor
        lr_monitor = LearningRateMonitor(logging_interval='step')
        self.callbacks.append(lr_monitor)

class BaseLitModule(pl.LightningModule, metaclass=abc.ABCMeta):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model
        self.val_results = None
        self.save_hyperparameters(copy.deepcopy(config))
        self.PrepareDataset()

        assert 'val_results_path' in self.config['exp_args']
        if not os.path.exists(self.config['exp_args']['val_results_path']):
            os.system('mkdir -p %s'%(self.config['exp_args']['val_results_path']))
    
    def PrepareDataset(self):
        config = self.config
        train_datafunc = Tools.rgetattr(Datasets, config['dataset_args']['train']['dataset_type'])
        train_data = train_datafunc(**config['dataset_args']['train'])
        val_datafunc = Tools.rgetattr(Datasets, config['dataset_args']['val']['dataset_type'])
        val_data = val_datafunc(**config['dataset_args']['val'])

        self.train_data = train_data
        self.val_data = val_data
    
    def WriteValResults(self, results):
        writer = self.logger.experiment
        for key, val in results.items():
            writer.add_scalar('Eval/%s'%(key), val, self.current_epoch)

    def train_dataloader(self):
        self.train_dataloader_obj = self.train_data.CreateLoader()
        return self.train_dataloader_obj
    
    def val_dataloader(self):
        self.val_dataloader_obj = self.val_data.CreateLoader()
        return self.val_dataloader_obj
    
    def configure_optimizers(self):
        optimizer_args = self.config['fitting_args']['optimizer_args']
        scheduler_args = self.config['fitting_args']['scheduler_args']
        optimizer_func = getattr(torch.optim, optimizer_args['type'])
        optimizer = optimizer_func(self.model.parameters(), **optimizer_args['args'])

        if scheduler_args is not None:
            scheduler_func = getattr(torch.optim.lr_scheduler, scheduler_args['type'])
            scheduler = scheduler_func(optimizer, **scheduler_args['args'])
            print (optimizer, scheduler)
            return [optimizer], [scheduler]
        else: 
            print (optimizer)
            return [optimizer]
    
    def on_train_epoch_start(self):
        if self.global_rank == 0: 
            print ('Epoch %d/%d'%(self.current_epoch, self.config['exp_args']['epoch']-1))

    def on_train_epoch_end(self):
        if self.global_rank == 0:
            if self.val_results is not None:
                print (self.val_results)
                self.WriteValResults(self.val_results)
                self.model.Save(self.current_epoch, accuracy=self.val_results['AP'], replace=True)
            else:
                print ('No val results!')
            self.val_results = None
    
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        idx = batch['idx']
        f_path = self.config['exp_args']['val_results_path'] + '/' + '%.5d.pkl'%idx
        torch.save(outputs, f_path)
    
    @abc.abstractmethod
    def validation_epoch_end(self, val_outs):
        """
        Must implement
        """
        raise NotImplementedError

    @abc.abstractmethod
    def training_step(self, batch, batch_idx):
        """
        Must implement
        """
    
    @abc.abstractmethod
    def validation_step(self, batch, batch_idx):
        """
        Must implement
        """

    @abc.abstractmethod
    def write_logger(self, *args, **kwargs):
        pass