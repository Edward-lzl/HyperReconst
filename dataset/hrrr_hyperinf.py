import pandas as pd
import os
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import xarray as xr
from easydict import EasyDict
import random
import io, pickle
import sys
import json
#import utils.mypath as mypath
import math
from multiprocessing import shared_memory,Pool
import multiprocessing as mp
import copy
import matplotlib.pyplot as plt
import queue,time,tqdm

DEM_MEAN = 284.592607421875
DEM_STD = 222.36515958509085

T_MEAN = 288.98371189102784
T_STD = 288.98371189102784

class HRRR_INF(torch.utils.data.Dataset):
    def __init__(self, train_config, type):
        super().__init__()
        self.train_cfg = train_config
        self.hrrr_data_path = train_config.hrrr_data_path
        self.weather5k_data_path = train_config.weather5k_data_path
        
        #self.hrrr_norm_path = train_config.norm_path.hrrr_norm_path
        #self.w5k_norm_path = train_config.norm_path.w5k_norm_path

        self.pred_names = self.train_cfg.pred_names

        self.target_resolution = train_config.target_resolution
        
        self.lon_range = train_config.lon_range
        self.lat_range = train_config.lat_range
        raw_hrrr_lon_range = np.arange(239., 286., 0.03125)
        raw_hrrr_lat_range = np.arange(25., 47., 0.03125)

        lon_grid, lat_grid = np.meshgrid(np.arange(self.lon_range[0],self.lon_range[1],0.03125),np.arange(self.lat_range[0], self.lat_range[1], 0.03125))
        self.lon_lat_grid = np.stack([lon_grid.ravel(), lat_grid.ravel()], axis=-1)
        
        self.lon_index_left = np.where(np.isclose(raw_hrrr_lon_range, self.lon_range[0]))[0][0]
        self.lon_index_right = np.where(np.isclose(raw_hrrr_lon_range, self.lon_range[1]))[0][0]
        self.lat_index_bottom = np.where(np.isclose(raw_hrrr_lat_range, self.lat_range[0]))[0][0]
        self.lat_index_top = np.where(np.isclose(raw_hrrr_lat_range, self.lat_range[1]))[0][0]

        self.type = type
        
        self.stn_data = np.load(self.weather5k_data_path+'/'+type+'_data_hrrr_r1.npy')
        
        self.dem_data = np.load('/home/bingxing2/ailab/scxlab0056/DATA/ele_hrrr_704x1504.npy')[self.lat_index_bottom:self.lat_index_top, self.lon_index_left:self.lon_index_right]
        # with open(self.w5k_norm_path, mode='r') as f:
        #     mean_std = json.load(f)
        
        self.dem_data = (self.dem_data - DEM_MEAN) / DEM_STD
        
        
        if self.target_resolution == 0.0625:
            self.dem_data = self.dem_data[::2,::2]
        
        if self.type == 'train':
            time_span = self.train_cfg.train_time_span
            with open('/home/bingxing2/ailab/scxlab0056/CODE/MambaDS/tools/hrrr_file_filter.json', 'r') as f:
                hrrr_filter = json.load(f)
        elif self.type == 'valid':
            time_span = self.train_cfg.valid_time_span
            with open('/home/bingxing2/ailab/scxlab0056/CODE/MambaDS/tools/valid_hrrr_file_filter.json', 'r') as f:
                hrrr_filter = json.load(f)
        elif self.type == 'test':
            time_span = self.train_cfg.test_time_span
            with open('/home/bingxing2/ailab/scxlab0056/CODE/MambaDS/tools/test_hrrr_file_filter.json', 'r') as f:
                hrrr_filter = json.load(f)
        else:
            raise NotImplementedError

        self.start_time = time_span[0]
        self.end_time = time_span[1]
        self.time_list = pd.date_range(self.start_time, self.end_time, freq=str(1)+'h')
            
        self.hrrr_files = self._get_hrrr_files(self.time_list, pred_names=self.train_cfg.pred_names)
        
        mask = np.ones(len(self.hrrr_files),dtype=bool)
        mask[hrrr_filter] = False
        self.hrrr_files = [i for index, i in enumerate(self.hrrr_files) if index not in hrrr_filter]
        # self.hrrr_files = self.hrrr_files[:100]
        self.stn_data = self.stn_data[:,mask,:]
        # self.stn_data = self.stn_data[:,:100,:]
        assert len(self.hrrr_files) == self.stn_data.shape[1]
        
        self.input_resolution = train_config.input_resolution
        self.scale_factor = int(self.input_resolution / self.target_resolution)
        
        self.hrrr_size_lon = int((self.lon_range[1]-self.lon_range[0])/self.target_resolution)
        self.hrrr_size_lat = int((self.lat_range[1]-self.lat_range[0])/self.target_resolution)
        
        
        # result_dict = self.__getitem__(0)
    
    def __getitem__(self, item):
        
        
        hrrr_idx = item % len(self.hrrr_files)

        hrrr_file = self.hrrr_files[hrrr_idx]
        
        hrrr_input =  self.get_hrrr_data(hrrr_file)[0]
        
        # import pdb
        # pdb.set_trace()
        
        
        stn_data = self.stn_data[:,hrrr_idx,:].copy()
        stn_data = torch.from_numpy(stn_data)
        
        stn_data[:,0] = stn_data[:,0] + 273.15
        stn_data[:,0] = (stn_data[:,0] - T_MEAN) / T_STD
        if stn_data[:,2].mean() < 0:
            stn_data[:,2] = 360 + stn_data[:,2]
        
        hrrr_interp = (self._interp_hrrr_to_stn(hrrr_input, stn_data[:,2:4]) - T_MEAN) / T_STD
        
        hrrr_input =  (self.get_hrrr_data(hrrr_file)[0] - T_MEAN) / T_STD
        
        hrrr_interp = torch.cat((hrrr_interp.unsqueeze(1), hrrr_input.view(1, -1).permute(1,0).contiguous()), dim=0)

        hrrr_data_label = hrrr_input
        
        
        
        coord_stn = stn_data[:,2:4]
        coord_grid = torch.tensor(self.lon_lat_grid, dtype=torch.float32)
        coord_input = torch.cat([coord_stn,coord_grid],dim=0)
        coord_input[:,0] = coord_input[:,0]/360.
        coord_input[:,1] = coord_input[:,1]/90.
        
        feat_stn = (stn_data[:,4].unsqueeze(1).float()-DEM_MEAN) / DEM_STD
        month = torch.full(feat_stn.shape, int(hrrr_file[0][-30:-28])/12)
        day = torch.full(feat_stn.shape, int(hrrr_file[0][-28:-26])/31)
        hour = torch.full(feat_stn.shape, int(hrrr_file[0][-17:-15])/24)
        res = torch.full(feat_stn.shape,0)
        feat_stn = torch.cat((month, day, hour, res, feat_stn),dim=1)
        feat_grid = torch.from_numpy(self.dem_data).unsqueeze(0).float().view(-1,1)
        res = torch.full(feat_grid.shape,0.03125)
        month = torch.full(feat_grid.shape, int(hrrr_file[0][-30:-28])/12)
        day = torch.full(feat_grid.shape, int(hrrr_file[0][-28:-26])/31)
        hour = torch.full(feat_grid.shape, int(hrrr_file[0][-17:-15])/24)
        feat_grid = torch.cat((month, day, hour, res, feat_grid),dim=1)
        feat = torch.cat((feat_stn,feat_grid),dim=0)
        
        label = torch.cat((stn_data[:,0].unsqueeze(1),hrrr_data_label.view(1,-1).permute(1,0).contiguous()), dim=0)
        

        return {
                'hrrr_input': hrrr_input.float(),
                'coords': coord_input.float(),
                'feat': feat.float(),
                'label': label.float(),
                'hrrr_interp': hrrr_interp.float(),
                'hrrr_file': hrrr_file
                }

    def _interp_hrrr_to_stn(self, hrrr_data, coord):
        
        _, H, W = hrrr_data.shape
        in_lon = self.lon_range[0] + np.array(range(W)) * self.input_resolution
        in_lat = self.lat_range[0] + np.array(range(H)) * self.input_resolution
        
        out_lon = coord[:,0]
        out_lat = coord[:,1]
        
        hrrr_data = xr.DataArray(data=hrrr_data[0,:,:], dims=['y','x'], coords=(in_lat.tolist(), in_lon.tolist()))
        interp_data = hrrr_data.interp(x=xr.DataArray(out_lon, dims='z'),y=xr.DataArray(out_lat, dims='z'))
        return torch.from_numpy(interp_data.data)
    
    def get_hrrr_data(self, hrrr_file):
        hrrr_data_surface = np.zeros((1, len(self.train_cfg.pred_names.hrrr), self.hrrr_size_lat, self.hrrr_size_lon))
        for i in range(len(hrrr_file)):
            data = np.load(hrrr_file[i])[:,:]
            
            if self.target_resolution == 0.03125:
                hrrr_data_surface[0, i, :, :] = data[self.lat_index_bottom:self.lat_index_top, self.lon_index_left:self.lon_index_right]
            elif self.target_resolution == 0.0625:
                hrrr_data_surface[0, i, :, :] = data[self.lat_index_bottom:self.lat_index_top:2, self.lon_index_left:self.lon_index_right:2]
        hrrr_data_surface = torch.from_numpy(hrrr_data_surface).float()
        
        return hrrr_data_surface


    def __len__(self):
        
        return len(self.hrrr_files)
        
    def _get_hrrr_files(self, time_list, pred_names):
        tmp_list = []
        for tmp_time in time_list:
            time_str = str(tmp_time.to_datetime64()).split('T')
            year = time_str[0].split('-')[0]
            month = time_str[0].split('-')[1]
            day = time_str[0].split('-')[2]
            
            url_list = [] 
            for var_name in pred_names['hrrr']:
                url = f"{self.hrrr_data_path}hrrr.{year}{month}{day}/{var_name}_hrrr.t{time_str[1][0:2]}z.wrfsfcf00.npy"
                url_list.append(url)
            tmp_list.append(url_list)
        return tmp_list
        

    def norm_data(self, data, data_type='h8', norm_type='mean_std', with_noise=False):
        
        if norm_type.lower() == 'mean_std':
            if data_type.lower() == 'h8':
                norm_dict = np.load(self.h8_norm_data_path, allow_pickle=True).item()
                mean = norm_dict['mean']
                mean = np.expand_dims(mean, axis=[0,2,3])
                std = norm_dict['std']
                std = np.expand_dims(std, axis=[0,2,3])
                data = (data - mean) / std
            elif data_type.lower() == 'pred':
                
                with open(self.pred_single_norm_path, mode='r') as f:
                    single_level_mean_std = json.load(f)

                if data.shape[1] == 4:
                    for idx, var_name in enumerate(self.train_cfg.pred_names.surface):
                        # [np.newaxis,:,np.newaxis,np.newaxis]
                        mean = np.array(single_level_mean_std['mean'][var_name])
                        std = np.array(single_level_mean_std['std'][var_name])
                        if with_noise:
                            noise_factor = {
                                'u10': 0.5,
                                'v10': 0.5,
                                't2m': 0.2,
                                'sp': 0.2,
                                'tp1h': 0.3
                            }
                            noise_std = noise_factor[var_name] * std
                            noise = np.random.normal(0, noise_std, data[:,idx,:,:].shape)
                            data[:,idx,:,:] += noise
                        data[:,idx,:,:] = (data[:,idx,:,:] - mean) / std
            elif data_type.lower() == 'hrrr':
                with open(self.hrrr_norm_path, mode='r') as f:
                    mean_std = json.load(f)
                    for idx, var_name in enumerate(self.train_cfg.pred_names.hrrr):
                        mean = np.array(mean_std['mean'][var_name])
                        std = np.array(mean_std['std'][var_name])
                        data[:,idx,:,:] = (data[:,idx,:,:] - mean) / std

        else:
            raise NotImplementedError

        return data
        


if __name__ == '__main__':
    config_path = '/home/bingxing2/ailab/scxlab0056/CODE/Reconst_DS/config/hrrr/hyper_inf.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)
    dataset = HRRR_INF(config.train_cfg, 'valid')
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle=True,
                                                                drop_last=False,num_workers=4,
                                                                pin_memory = True, prefetch_factor = 3,  persistent_workers = True)

    for epoch in range(3):
        # if epoch == 1:
            import pdb
            pdb.set_trace()
            list = []
            for i, data in enumerate(train_dataloader):
                hrrr_data = data['hrrr_input']
                coord_data = data['coords']
                feat_data = data['feat']
                state = data['hrrr_interp']
                label = data['label']
                error = (hrrr_data*T_STD+T_MEAN)[0].mean()-(label[:,:-320*320,:]*T_STD+T_MEAN)[0].mean()
                list.append(error)
                #print((hrrr_data*T_STD+T_MEAN)[0].mean()-(label[:,:-320*320,:]*T_STD+T_MEAN)[0].mean())
                # print()
            print(max(list),np.mean(list))
        
