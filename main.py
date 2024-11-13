import argparse
import os
import yaml
from easydict import EasyDict
from reconst_hrrr import Reconst_HRRR

def parse_args():
    parser = argparse.ArgumentParser(
        description = 'Pytorch implementation of Data Assimulation with Remote Sensing by Edward Liu'
    )
    parser.add_argument('--config', default = '/mnt/petrelfs/liuzili/code/DS_ISPRS/configs/rs2era.yaml')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    for k, v in vars(args).items():
        config[k] = v
    config = EasyDict(config)
    
    operator = Reconst_HRRR(config)
    
    if config.mode == 'train':
        operator.train()
    elif config.mode == 'test':
        operator.inference()

if __name__ == '__main__':
    main()
    print("FIN")
