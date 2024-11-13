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

class HRRR_Dataset(torch.utils.data.Dataset):
    def __init__(self, train_config, type):
        super().__init__()
        self.train_cfg = train_config
        self.hrrr_data_path = train_config.hrrr_data_path
        self.weather5k_data_path = train_config.weather5k_data_path
        
        self.hrrr_norm_path = train_config.norm_path.hrrr_norm_path
        self.w5k_norm_path = train_config.norm_path.w5k_norm_path

        self.pred_names = self.train_cfg.pred_names

        self.target_resolution = train_config.target_resolution
        
        self.lon_range = train_config.lon_range
        self.lat_range = train_config.lat_range
        raw_hrrr_lon_range = np.arange(239., 286., 0.03125)
        raw_hrrr_lat_range = np.arange(25., 47., 0.03125)
        self.lon_index_left = np.where(np.isclose(raw_hrrr_lon_range, self.lon_range[0]))[0][0]
        self.lon_index_right = np.where(np.isclose(raw_hrrr_lon_range, self.lon_range[1]))[0][0]
        self.lat_index_bottom = np.where(np.isclose(raw_hrrr_lat_range, self.lat_range[0]))[0][0]
        self.lat_index_top = np.where(np.isclose(raw_hrrr_lat_range, self.lat_range[1]))[0][0]
        self.type = type
        self.stn_data = np.load(self.weather5k_data_path+'/'+type+'_data_hrrr_r1.npy')
        
        self.dem_data = np.load('/home/bingxing2/ailab/scxlab0056/DATA/ele_hrrr_704x1504.npy')[self.lat_index_bottom:self.lat_index_top, self.lon_index_left:self.lon_index_right]
        with open(self.w5k_norm_path, mode='r') as f:
            mean_std = json.load(f)
        
        if train_config.with_norm:
            mean = np.array(mean_std['mean']['ele'])
            std = np.array(mean_std['std']['ele'])
            self.dem_data = (self.dem_data - mean) / std
        
        
        if self.target_resolution == 0.0625:
            self.dem_data = self.dem_data[::2,::2]
        if train_config.with_norm:
            with open(self.w5k_norm_path, mode='r') as f:
                mean_std = json.load(f)
                for idx, var_name in enumerate(self.train_cfg.pred_names.w5k):
                    mean = np.array(mean_std['mean'][var_name])
                    std = np.array(mean_std['std'][var_name])
                    self.stn_data[:,:,idx] = (self.stn_data[:,:,idx] - mean) / std
        
        if self.type == 'train':
            time_span = self.train_cfg.train_time_span
            with open('/home/bingxing2/ailab/scxlab0056/CODE/MambaDS/tools/hrrr_file_filter.json', 'r') as f:
                hrrr_filter = json.load(f)
        elif self.type == 'val':
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
        self.hrrr_files = [item for index, item in enumerate(self.hrrr_files) if index not in hrrr_filter]
      
        self.stn_data = self.stn_data[:,mask,:]
        
        self.input_resolution = train_config.input_resolution
        self.scale_factor = int(self.input_resolution / self.target_resolution)
        
        self.hrrr_size_lon = int((self.lon_range[1]-self.lon_range[0])/self.target_resolution)
        self.hrrr_size_lat = int((self.lat_range[1]-self.lat_range[0])/self.target_resolution)
        if train_config.with_norm:
            self.has_normed = False
        else:
            self.has_normed = True
        
        result_dict = self.__getitem__(1000)
        
    
    def load_data_process(self):
        while True:
            job_pid, file, frame_id = self.index_queue.get()
            if job_pid not in self.compound_data_queue_dict:
                try:
                    self.lock.acquire()
                    for i in range(self.compound_data_queue_num):
                        if job_pid == self.arr[i]:
                            self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                            self.sharedmemory_dict[job_pid] = self.sharedmemory_list[i]
                            break
                    if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                        print("error", job_pid, self.arr)
                except Exception as err:
                    raise err
                finally:
                    self.lock.release()
            
            b = np.ndarray(self.h8.shape, dtype=self.h8.dtype, buffer=self.sharedmemory_dict[job_pid].buf)
            # start_time = time.time()
            #with io.BytesIO(self.client.get(file)) as f:
            unit_data = np.load(file)
            # end_time = time.time()
            # print("h8_laoding time:", end_time-start_time)
            b[frame_id] = unit_data
            self.unit_data_queue.put((job_pid, file, frame_id))
    
    def data_compound_process(self):
        recorder_dict = {}
        while True:
            job_pid, file, frame_id = self.unit_data_queue.get()
            if job_pid not in self.compound_data_queue_dict:
                try:
                    self.lock.acquire()
                    for i in range(self.compound_data_queue_num):
                        if job_pid == self.arr[i]:
                            self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                            break
                    if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                        print("error", job_pid, self.arr)
                except Exception as err:
                    raise err
                finally:
                    self.lock.release()
            if (job_pid) in recorder_dict:
                recorder_dict[(job_pid)][(file, frame_id)] = 1
            else:
                recorder_dict[(job_pid)] = {(file, frame_id): 1}
            # print("recorder_len", len(recorder_dict[job_pid]))
            # print("recorder_dict", recorder_dict)
            if len(recorder_dict[job_pid]) == self.data_element_num:
                del recorder_dict[job_pid]
                self.compound_data_queue_dict[job_pid].put((file))
    
    def get_data(self, files, type):
        
        job_pid = os.getpid()
        
        if job_pid not in self.compound_data_queue_dict:
            try:
                self.lock.acquire()
                for i in range(self.compound_data_queue_num):
                    if i == self.arr[i]:
                        self.arr[i] = job_pid
                        self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                        self.sharedmemory_dict[job_pid] = self.sharedmemory_list[i]
                        break
                if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                    print("error", job_pid, self.arr)

            except Exception as err:
                raise err
            finally:
                self.lock.release()
        try:
            file = self.compound_data_queue_dict[job_pid].get(False)
            raise ValueError
        except queue.Empty:
            pass
        except Exception as err:
            raise err
        if type == 'h8':
            b = np.ndarray(self.h8.shape, dtype=self.h8.dtype, buffer=self.sharedmemory_dict[job_pid].buf)
            for frame_id, file in enumerate(files[:4]):
                self.index_queue.put((job_pid, file, frame_id))
            file = self.compound_data_queue_dict[job_pid].get()
            data_tmp = copy.deepcopy(b)
            data_tmp[np.isnan(data_tmp)] = 0
            if (not self.has_normed):
                data_tmp = self.norm_data(data_tmp.reshape(1,4,720,1120), data_type='h8')
            
            data_tmp = torch.from_numpy(data_tmp).float().contiguous()

        elif type == 'era5':
            pass
        else:
            raise NotImplementedError
        return data_tmp
    
    def downsample(self, data, scale, type='direct'):
        if type == 'direct':
            data = data[:,:,::scale,::scale].contiguous()
        elif type == 'avgpool':
            out_h = math.ceil(data.shape[2] / scale)
            out_w = math.ceil(data.shape[3] / scale)
            data = F.interpolate(data, size=(out_h, out_w), mode='bilinear').contiguous()
        else:
            raise NotImplementedError
        return data

    
    def __getitem__(self, item):

        hrrr_idx = item % len(self.hrrr_files)

        hrrr_file = self.hrrr_files[hrrr_idx]

        hrrr_data_label = self.get_hrrr_data(hrrr_file)
        
        
        stn_data = self.stn_data[:,hrrr_idx,:]
        stn_data = torch.from_numpy(stn_data)
        stn_data[:,0] = stn_data[:,0] + 273.15
        if stn_data[:,2].mean() < 0:
            stn_data[:,2] = 360 + stn_data[:,2]
        #print(hrrr_file)
        import pdb
        
        pdb.set_trace()
        #plt.imshow(hrrr_data_label[0,0,:,:].numpy()[::-1,:], extent=[275,285,35,45],cmap='RdYlBu_r') 
        #plt.scatter(stn_data[:,2].numpy().tolist(),stn_data[:,3].numpy().tolist(),c=stn_data[:,0].numpy(),edgecolors='k',cmap='RdYlBu_r')
        #plt.savefig('plot.png', dpi=300, bbox_inches='tight')
        #self.get_era5_data(era5_file,with_noise=True)
        
        aux_data = torch.from_numpy(self.dem_data).unsqueeze(0).float()

        return {
                # 'h8_data': h8_data[0], 'h8_delta_t': h8_delta_t[0], 
                'hrrr_file': hrrr_file,
                'hrrr_data': hrrr_data_label[0], 'stn_data': stn_data, 
                'aux_data': aux_data
                }

    def get_hrrr_data(self, hrrr_file):
        hrrr_data_surface = np.zeros((1, len(self.train_cfg.pred_names.hrrr), self.hrrr_size_lat, self.hrrr_size_lon))
        for i in range(len(hrrr_file)):
            data = np.load(hrrr_file[i])[:,:]
            
            if self.target_resolution == 0.03125:
                hrrr_data_surface[0, i, :, :] = data[self.lat_index_bottom:self.lat_index_top, self.lon_index_left:self.lon_index_right]
            elif self.target_resolution == 0.0625:
                hrrr_data_surface[0, i, :, :] = data[self.lat_index_bottom:self.lat_index_top:2, self.lon_index_left:self.lon_index_right:2]
        if (not self.has_normed):
            hrrr_data_surface = self.norm_data(hrrr_data_surface, data_type='hrrr')
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
    config_path = '/home/bingxing2/ailab/scxlab0056/CODE/Reconst_DS/config/hrrr/demo.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)
    dataset = HRRR_Dataset(config.train_cfg, 'train')

            
