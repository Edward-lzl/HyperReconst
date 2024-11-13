import sys
sys.path.append('/home/bingxing2/ailab/scxlab0056/CODE/Reconst_DS')
from easydict import EasyDict
import yaml
#from models.baselines.variable_net import VariableNet
from models.backbone.builder import build_backbone
import torch
import torch.nn as nn
import torch.nn.functional as F

from numpy import pi

class ContiReconstNet(nn.Module):
    def __init__(self,
                 encoder_cfg: dict,
                 variable_net_cfg: dict,
                 **kwargs):
        super().__init__()
        self.encoder = build_backbone(**encoder_cfg)
        #self.variable_net = VariableNet(**variable_net_cfg)
    
    def _forward_single(self, field_feature, coord, dx, dy, ele):
        #output = self.variable_net(field_feature, coord, dx, dy, ele)
        #return output
        pass
    
    def forward(self, fields, coord, dx, dy, ele, **kwargs):
        _, endpoints = self.encoder(fields)
        import pdb
        pdb.set_trace() 
        output = self._forward_single(endpoints['block4'], coord, dx, dy, ele)
        
        return output
        
if __name__ == '__main__':
    config_path = '/home/bingxing2/ailab/scxlab0056/CODE/Reconst_DS/config/hrrr/demo.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)
    
    model = ContiReconstNet(encoder_cfg=config.network.encoder_cfg, variable_net_cfg=config.network.variable_net_cfg)
    field = torch.randn((1, 1, 320, 320))
    output = model(field,0,0,0,0)
   