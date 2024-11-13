import numpy as np
import torch
import torch.utils.data as data
from torch import nn, optim
import torch.utils.checkpoint as checkpoint
from torch.utils.tensorboard import SummaryWriter
import logging
from dataset.builder import build_dataset
from models.builder import build_model
import xarray as xr
import os
import yaml
from easydict import EasyDict
import json
import glob
import time
#from utils.time_metric import TimeMetric
import tqdm
import shutil
import torch.nn.functional as F
import sys
#from utils.positional_encoding import SineCosPE
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs

from tools.vis_field import VisField
'''
from src.metrics import psnr
from src.myloss import CharbonnierLoss

from pytorch_msssim import ssim
'''

T_MEAN = 288.98371189102784
T_STD = 288.98371189102784

class Reconst_HRRR():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.exp_path = os.path.join(self.config.exp_parent_path, self.config.exp_name)
        self.log_path = os.path.join(self.exp_path,'logs')
        os.makedirs(self.log_path, exist_ok=True)
        
        self.lon_range = config.train_cfg.lon_range
        self.lat_range = config.train_cfg.lat_range
        self.input_resolution = config.train_cfg.input_resolution
        self.target_resolution = config.train_cfg.target_resolution
        
        self.accelerator = Accelerator(
                                    kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False,find_unused_parameters=True)] #,find_unused_parameters=True
                                    )
        self.device = self.accelerator.device
        
        self.VisUtil = VisField(self.lon_range, self.lat_range)
        
        self.vis_path = os.path.join(self.log_path, 'vis')
        os.makedirs(self.vis_path, exist_ok=True)
        
        self._build()
    
    def _build(self):
        self._build_dir()
        self._build_data_loader()
        self._build_model()
        self._build_optimizer()
        
    def train_one_epoch(self, epoch):
        train_loss_dict = {'pred_grid_loss': 0,
                           'interp_grid_loss': 0,
                           'pred_stn_loss': 0,
                           'interp_stn_loss': 0}
        
        for i, data in enumerate(self.train_dataloader):
            if i >= 2 and (self.glob_step-1) % self.log_step == 0:
                end_time = time.time()
                iter_time = (end_time - start_time) 
                self.accelerator.print(f"[Epoch:{epoch}/{self.num_epoch}][batch:{i}/{len(self.train_dataloader)}]: train_loss_grid:{loss_grid.item()}/train_stn_loss:{loss_stn.item()}/interp_stn_loss:{loss_stn_interp.item()}/speed{iter_time}s")
            start_time = time.time()
            self.glob_step = self.glob_step + 1
            self.model.train()
            
            hrrr_data = data['hrrr_input'].to(self.device)
            coord_data = data['coords'].to(self.device)
            feat_data = data['feat'].to(self.device)
            state = data['hrrr_interp'].to(self.device)
            label = data['label'].to(self.device)
            
            
            output_mlp, output_grid = self.model.forward(hrrr_data, state, coord_data, feat_data)
            
            interp_stn = self._interp_hrrr_to_stn(hrrr_data, coord_data)
            
            result_grid = output_mlp[:,-output_grid.shape[2]*output_grid.shape[3]:,:] + \
                            hrrr_data.view(output_grid.shape[0],output_grid.shape[1],-1).permute(0,2,1).contiguous()
            loss_grid = self.criterion(label[:,-output_grid.shape[2]*output_grid.shape[3]:,:]*T_STD+T_MEAN, result_grid*T_STD+T_MEAN)

            result_stn = output_mlp[:,:-output_grid.shape[2]*output_grid.shape[3]] + \
                            interp_stn.permute(0,2,1).contiguous()
            loss_stn = self.criterion(label[:,:-output_grid.shape[2]*output_grid.shape[3],:]*T_STD+T_MEAN, result_stn*T_STD+T_MEAN)
            
            loss = 1*loss_grid + loss_stn
            
            train_loss_dict['pred_grid_loss'] += loss_grid
            train_loss_dict['pred_stn_loss'] += loss_stn
            
            loss_stn_interp = self.criterion(label[:,:-output_grid.shape[2]*output_grid.shape[3],:]*T_STD+T_MEAN, interp_stn.permute(0,2,1).contiguous()*T_STD+T_MEAN)
            train_loss_dict['interp_stn_loss'] += loss_stn_interp
            #print(loss_stn_interp)
            
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()
            torch.cuda.empty_cache()
            
            if (self.glob_step - 1) % (self.log_step*10) == 0:
                if self.accelerator.is_main_process:
                    result_file_path = os.path.join(self.vis_path, 'train_vis')
                    os.makedirs(result_file_path, exist_ok=True)
                    result_file = os.path.join(result_file_path, f'{self.glob_step}_t2m.png')
                    self.VisUtil.forward_single_image_w_stn(result_grid[0], 
                                                            result_stn[0],
                                                            interp_stn[0],
                                                            label[0,-output_grid.shape[2]*output_grid.shape[3]:,:],
                                                            label[0,:-output_grid.shape[2]*output_grid.shape[3],:],
                                                            coord_data[0],
                                                            't2m',
                                                            result_file)
            
        return train_loss_dict
    
    def valid_one_epoch(self, epoch):
        valid_loss_dict = {'pred_grid_loss': 0,
                        'interp_grid_loss': 0,
                        'pred_stn_loss': 0,
                        'interp_stn_loss': 0}
        
        with torch.no_grad():
            for i, data in (enumerate(self.valid_dataloader)):
                hrrr_data = data['hrrr_input'].to(self.device)
                coord_data = data['coords'].to(self.device)
                feat_data = data['feat'].to(self.device)
                state = data['hrrr_interp'].to(self.device)
                label = data['label'].to(self.device)
                
                output_mlp, output_grid = self.model.forward(hrrr_data, state, coord_data, feat_data)
            
                interp_stn = self._interp_hrrr_to_stn(hrrr_data, coord_data)
                
                result_grid = output_mlp[:,-output_grid.shape[2]*output_grid.shape[3]:,:] + \
                                hrrr_data.view(output_grid.shape[0],output_grid.shape[1],-1).permute(0,2,1).contiguous()
                loss_grid = self.criterion(label[:,-output_grid.shape[2]*output_grid.shape[3]:,:]*T_STD+T_MEAN, result_grid*T_STD+T_MEAN)

                result_stn = output_mlp[:,:-output_grid.shape[2]*output_grid.shape[3]] + \
                                interp_stn.permute(0,2,1).contiguous()
                loss_stn = self.criterion(label[:,:-output_grid.shape[2]*output_grid.shape[3],:]*T_STD+T_MEAN, result_stn*T_STD+T_MEAN)

                valid_loss_dict['pred_grid_loss'] += loss_grid
                valid_loss_dict['pred_stn_loss'] += loss_stn
                
                loss_stn_interp = self.criterion(label[:,:-output_grid.shape[2]*output_grid.shape[3],:]*T_STD+T_MEAN, interp_stn.permute(0,2,1).contiguous()*T_STD+T_MEAN)
                valid_loss_dict['interp_stn_loss'] += loss_stn_interp
                
        return valid_loss_dict
    
    def train(self):
        self.batch_size = self.config.train_cfg.batch_size
        self.num_epoch = self.config.train_cfg.num_epoch
        self.log_step = self.config.train_cfg.log.log_step
        self.model.to(self.device)
        
        self.criterion = self._build_loss()
        
        lr = self.optimizer.param_groups[0]['lr']
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(self.model, self.optimizer, self.train_dataloader)
        best_loss = 100.
        for epoch in range(self.epoch, self.num_epoch):
            
            self.model.train()
            self.accelerator.print(f"Lr at epoch {epoch} is: {self.optimizer.param_groups[0]['lr']}")
            train_loss_dict = self.train_one_epoch(epoch)
            self.accelerator.print(f"=>[Epoch:{epoch}/{self.num_epoch}]")
            self.accelerator.print(f"==>[train_loss]:\npred_stn_loss:{train_loss_dict['pred_stn_loss']/len(self.train_dataloader)}/interp_stn_loss:{train_loss_dict['interp_stn_loss']/len(self.train_dataloader)}")
            
            self.model.eval()
            valid_loss_dict = self.valid_one_epoch(epoch)
            self.accelerator.print(f"==>[valid_loss]:\npred_stn_loss:{valid_loss_dict['pred_stn_loss']/len(self.valid_dataloader)}/interp_stn_loss:{valid_loss_dict['interp_stn_loss']/len(self.valid_dataloader)}")
            
            if self.accelerator.is_main_process:
                self.log.info(f"=>[Epoch:{epoch}/{self.num_epoch}]")
                self.log.info(f"==>[train_loss]:\npred_stn_loss:{train_loss_dict['pred_stn_loss']/len(self.train_dataloader)}/interp_stn_loss:{train_loss_dict['interp_stn_loss']/len(self.train_dataloader)}")
            
                self.log.info(f"==>[valid_loss]:\npred_stn_loss:{valid_loss_dict['pred_stn_loss']/len(self.valid_dataloader)}/interp_stn_loss:{valid_loss_dict['interp_stn_loss']/len(self.valid_dataloader)}")
            
            if valid_loss_dict['pred_stn_loss']/len(self.valid_dataloader) <= best_loss:
                best_loss = valid_loss_dict['pred_stn_loss']/len(self.valid_dataloader)
                best_flag = True
            else:
                best_flag = False
                
            if self.accelerator.is_main_process:
                if 'lr_schedule' in self.config.train_cfg.keys():
                    self.lr_schedule.step()
                #if best_flag:
                lr = self.optimizer.param_groups[0]['lr']
                if self.accelerator.is_main_process:
                    self.log_writer.add_scalar('learning_rate', lr, self.glob_step)
                state_dict = {}
                if isinstance(self.model, nn.Module):
                    state_dict['model'] = self.model.state_dict()
                else:
                    state_dict['model'] = self.model
                state_dict['epoch'] = epoch
                state_dict['global_step'] = self.glob_step

                model_file = os.path.join(self.checkpoint_path, f"{self.config.model_name}_epoch_latest.pth")
                unwarpped_model = self.accelerator.unwrap_model(self.model)
                state_dict['model'] = unwarpped_model
                self.accelerator.save(state_dict, model_file)
                if best_flag:
                    shutil.copy(model_file, os.path.join(self.checkpoint_path, 'best.pth'))
    
    def _build_optimizer(self):
        optim_dict = {'Adam': optim.Adam,
                      'SGD': optim.SGD}
        lr_schedule_dict = {'stepLR': optim.lr_scheduler.StepLR,
                            'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR,
                            'MultiStepLR': torch.optim.lr_scheduler.MultiStepLR}
        
        if self.config.train_cfg.optimizer.name in optim_dict.keys():
            self.optimizer = optim_dict[self.config.train_cfg.optimizer.pop('name')]([{'params': self.model.parameters(),
                                                                                **self.config.train_cfg.optimizer,
                                                                                'initial_lr': self.config.train_cfg.optimizer.lr}])
                
        else:
            raise NotImplementedError(f'Optimizer name {self.config.train_cfg.optimizer.name} not in optim_dict')
        
        if self.config.train_cfg.lr_schedule.name in lr_schedule_dict.keys():
            self.lr_schedule = lr_schedule_dict[self.config.train_cfg.lr_schedule.pop('name')](optimizer = self.optimizer,
                                                                                                  **self.config.train_cfg.lr_schedule, 
                                                                                                  last_epoch = self.epoch - 1)
        else:
            raise NotImplementedError(f'lr_schedule name {self.config.train_cfg.lr_schedule.name} not in lr_schedule_dict')
    
    def _build_model(self):
        self.accelerator.print("===> Initializing Model......")
        self.model = build_model(self.config.model_name, self.config.network)
        self.accelerator.print("===> Model Parameter Num:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        if self.config.train_cfg.resume or self.config.test_cfg.resume:
            if self.config.mode == 'train':
                checkpoint_cfg = self.config.train_cfg.checkpoint
            elif self.config.mode == 'test':
                checkpoint_cfg = self.config.test_cfg
            if checkpoint_cfg.checkpoint_name is None:
                model_file = os.path.join(self.checkpoint_path, checkpoint_cfg.checkpoint_name)
            else:
                # model_file = os.path.join(self.checkpoint_path, f"{self.config.exp_name}_latest.pth")
                model_file = checkpoint_cfg.checkpoint_name
            
            if not os.path.exists(model_file):
                self.accelerator.print(f"Warning: resume file {model_file} does not exist!")
                self.epoch, self.glob_step = 0, 0
            else:
                self.accelerator.print(f"Start to resume from {model_file}")
                state_dict = torch.load(model_file)
                
                try:
                    self.glob_step = state_dict.pop('global_step')
                except KeyError:
                    self.accelerator.print("Warning: global_step not in state dict!")
                    self.glob_step = 0
                try:
                    self.epoch = state_dict.pop('epoch') + 1
                except KeyError:
                    self.accelerator.print("Warning: epoch not in state dict!")
                    self.epoch = 0
                self.accelerator.print(f"===> Resume form epoch {self.epoch} global step {self.glob_step} model file {model_file}")
                if 'model' in state_dict.keys():
                    self.model.load_state_dict(state_dict['model'].state_dict(), strict=True)

                    # self.model.load_state_dict(state_dict['model'], strict=True)
                else:
                    self.model.load_state_dict(state_dict, state_dict=True)
        else:
            self.accelerator.print("===> Initialized model, training model from scratch!")
            self.epoch = 0
            self.glob_step =0
        
    def _build_dir(self):
        # logger
        
        if self.accelerator.is_main_process:
            log_name = f"{time.strftime('%Y-%m-%d-%H-%M')}.log"
            log_name = f"{self.config.exp_name}_{log_name}"
            log_dir = os.path.join(self.log_path, log_name)
            self.log = logging.getLogger()
            self.log.setLevel(logging.INFO)
            handler = logging.FileHandler(log_dir)
            self.log.addHandler(handler)
            self.log_writer = SummaryWriter(log_dir= self.log_path)

            self.log.info("Config:")
            self.log.info(self.config)
            self.log.info("\n")
        self.accelerator.print("Config:", self.config)

        self.checkpoint_path = os.path.join(self.config.exp_parent_path, self.config.exp_name+'/checkpoint')
        os.makedirs(self.checkpoint_path, exist_ok=True)
    
    def _build_data_loader(self):
        self.accelerator.print("===> Loading dataloader......")
        if self.config.mode == 'train':
            self.train_dataset = build_dataset(self.config.dataset_name, self.config.train_cfg, 'train')

            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size = self.config.train_cfg.batch_size, shuffle=True,
                                                                drop_last=False,num_workers=self.config.train_cfg.num_workers,
                                                                pin_memory = True, prefetch_factor = 3, persistent_workers = True)

            self.valid_dataset = build_dataset(self.config.dataset_name, self.config.train_cfg, 'valid')

            self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset, batch_size = self.config.train_cfg.batch_size, shuffle=False,
                                                                drop_last=False,num_workers=self.config.train_cfg.num_workers,
                                                                pin_memory = True, prefetch_factor = 3, persistent_workers = True)
        elif self.config.mode == 'test':
            self.test_dataset = build_dataset(self.config.dataset_name, self.config.train_cfg, 'test')
            self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size = self.config.train_cfg.batch_size, shuffle=False,
                                                                drop_last=False,num_workers=self.config.train_cfg.num_workers,
                                                                pin_memory = True, prefetch_factor = 3, persistent_workers = True)
        self.accelerator.print("===> Loaded dataloader !")
    
    def _build_loss(self):
        losses_dict = {'CrossEntropyLoss': nn.CrossEntropyLoss,
                       'L1Loss': nn.L1Loss,
                       'MSELoss': nn.MSELoss,
                       }
        loss_name = self.config.train_cfg.loss_cfg.name
        if 'scale' in self.config.train_cfg.loss_cfg.keys():
            self.scale = self.config.train_cfg.loss_cfg.scale
        if loss_name in losses_dict.keys():
            return losses_dict[loss_name]()
        else:
            raise NotImplementedError(f'Name {loss_name} not in losses_dcit')
    
    def _interp_hrrr_to_stn(self, hrrr_data, coord):
        
        hrrr_data = hrrr_data.detach().cpu().numpy()
        BSize, var_num, H, W = hrrr_data.shape
        in_lon = self.lon_range[0] + np.array(range(W)) * self.input_resolution
        in_lat = self.lat_range[0] + np.array(range(H)) * self.input_resolution
        
        in_lon = in_lon/360
        in_lat = in_lat/90
        
        out_lon = coord[0,:-H*W,0].cpu()
        out_lat = coord[0,:-H*W,1].cpu()
        
        
        
        hrrr_data = xr.DataArray(data=hrrr_data, dims=['bs', 'var', 'y','x'], coords=(np.array(range(BSize)).tolist(), np.array(range(var_num)), in_lat.tolist(), in_lon.tolist()))
        interp_data = hrrr_data.interp(x=xr.DataArray(out_lon.numpy(), dims='z'),y=xr.DataArray(out_lat.numpy(), dims='z'))
        return torch.from_numpy(interp_data.data).float().to(self.device)
    
    def inference_interp(self):
        loss_dict = {
                    'mse':
                            {'t': 0},
                    'mae':
                            {'t': 0}
                    }
        
        progress_bar = tqdm.tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))  
        T_MEAN = 288.98371189102784
        T_STD = 288.98371189102784
        for i, data in progress_bar:
            hrrr_data = data['hrrr_input'].to(self.device)*T_STD + T_MEAN
            coord_data = data['coords'].to(self.device)
            feat_data = data['feat'].to(self.device)
            state = data['hrrr_interp'].to(self.device)*T_STD + T_MEAN
            label = data['label'].to(self.device)*T_STD + T_MEAN
            hrrr_file = data['hrrr_file']
            #pdb.set_trace()
            import pdb
            # pdb.set_trace()
            
            #interp_data = self._interp_hrrr_to_stn(hrrr_data, coord_data)
            
            loss_dict['mse']['t'] += F.mse_loss(label[:,:-320*320,0], state[:,:-320*320,0])
            loss_dict['mae']['t'] += F.l1_loss(label[:,:-320*320,0], state[:,:-320*320,0])
            file_name = hrrr_file[0][0][-34:-15].replace('/','_')
            if self.accelerator.is_main_process:
                result_file_path = os.path.join(self.vis_path, 'train_vis')
                os.makedirs(result_file_path, exist_ok=True)
                result_file = os.path.join(result_file_path, f'{file_name}_t2m.png')
                self.VisUtil.forward_single_image_w_stn_gt(0, 
                                                        0,
                                                        0,
                                                        label[0,-320*320:,:],
                                                        label[0,:-320*320,:],
                                                        coord_data[0],
                                                        't2m',
                                                        result_file)
        
            
        for key, inner_dict in loss_dict.items():
            # 遍历内层字典
            for inner_key in inner_dict:
                
                inner_dict[inner_key] /= len(self.train_dataloader)
        print(loss_dict)
            

if __name__=='__main__':
    config_path = '/home/bingxing2/ailab/scxlab0056/CODE/Reconst_DS/config/hrrr/hyper_inf.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)
    operator = Reconst_HRRR(config)
    operator.train()
    