import sys
sys.path.append('/home/bingxing2/ailab/scxlab0056/CODE/Reconst_DS')
from models.baselines.baseline import ContiReconstNet
from models.hyper_inf import CombinedModel
import yaml
import xarray as xr
from easydict import EasyDict

model_dict = {
              'Baseline': ContiReconstNet,
              'HyperINF': CombinedModel
                }
'''
              'VSSM': VSSM,
              'VSSM_V1': VSSM_V1,
              'VSSM_S': VSSM_S,
              'VSSM_T': VSSM_T,
              'VSSM_T1': VSSM_T1,
              '''

def build_model(name='Baseline', network_config=None):
    if name in model_dict.keys():
        return model_dict[name](**network_config)
    else:
        raise NotImplementedError(r"{} is not an avaliable values in model_dict. ".format(name))


if __name__ == '__main__':
    config_path = '/home/bingxing2/ailab/scxlab0056/CODE/DS_ISPRS/configs/era5_sr_swin_mam.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)
    model = build_model(name=config.model_name, network_config=config.network)
    
