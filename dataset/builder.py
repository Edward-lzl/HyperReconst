import sys
sys.path.append('/home/bingxing2/ailab/scxlab0056/CODE/Reconst_DS')
# from dataset.dars_grid_dataset import DARS_Grid_Dataset
#from dataset.h8_era5_dataset import H8_ERA5
from dataset.hrrr_hyperinf import HRRR_INF
import yaml
import xarray as xr
from easydict import EasyDict
# 'DARS_Grid_Dataset' : DARS_Grid_Dataset,
dataset_dict = {
                'HRRR_Dataset': HRRR_INF
                }

def build_dataset(name='HRRR_Dataset', config = None, type = None):
    if name in dataset_dict.keys():
        return dataset_dict[name](config, type)
    else:
        raise NotImplementedError(r"{} is not an avaliable values in dataset_dict. ".format(name))


if __name__ == '__main__':
    config_path = '/home/bingxing2/ailab/scxlab0056/CODE/Reconst_DS/config/hrrr/demo.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)
    dataset = build_dataset(config.dataset_name, config.train_cfg, 'train')
