import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from typing import List, Optional, Dict, Tuple
import os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
torch.set_num_threads(1)

class ResMLP(nn.Module):
    def __init__(self,in_channels):
        super(ResMLP, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels),
                              # nn.Dropout(p=0.5),
                              nn.ReLU(inplace = True),
                              nn.Linear(in_channels, in_channels),
                              # nn.BatchNorm1d(in_channels)
                              )
    def forward(self,x):
        out = self.fc(x)
        return out + x

class BasicHyperNet(nn.Module):
    def __init__(self, input_channels: int, height: int, width: int):
        super().__init__()
        self.input_channels = input_channels
        self.height = height
        self.width = width
        #self.output_dim = output_dim
    
    def forward(self, x):
        raise NotImplementedError
    
class UNetDecodeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int = 0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class ResNet18UNetHyperNet(BasicHyperNet):
    def __init__(self, 
                 input_channels: int, 
                 height: int, 
                 width: int, 
                 upsample_factor: int = 1,  # 上采样倍数
                 output_layer: int = 4,     # 输出层索引 (0-4)
                 decoder_channels: List[int] = [256, 128, 64, 32, 3],
                 var_num=1):
        super().__init__(input_channels, height, width)
        
        assert -1 <= output_layer <= 4, "output_layer must be between 0 and 4"
        self.output_layer = output_layer
        self.upsample_factor = upsample_factor
        
        # 计算上采样后的输入尺寸
        self.upsampled_height = height * upsample_factor
        self.upsampled_width = width * upsample_factor
        
        resnet = timm.create_model('resnet18', pretrained=False)
        
        if input_channels != 3:
            self.encoder0 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.encoder0 = resnet.conv1
            
        self.encoder0_bn = resnet.bn1
        self.encoder0_pool = resnet.maxpool
        
        self.encoder1 = resnet.layer1  # 64 channels
        self.encoder2 = resnet.layer2  # 128 channels
        self.encoder3 = resnet.layer3  # 256 channels
        self.encoder4 = resnet.layer4  # 512 channels
        
        # Decoder layers with skip connections
        self.decoder4 = UNetDecodeBlock(512, decoder_channels[0])
        self.decoder3 = UNetDecodeBlock(decoder_channels[0], decoder_channels[1], skip_channels=256)
        self.decoder2 = UNetDecodeBlock(decoder_channels[1], decoder_channels[2], skip_channels=128)
        self.decoder1 = UNetDecodeBlock(decoder_channels[2], decoder_channels[3], skip_channels=64)
        self.decoder0 = UNetDecodeBlock(decoder_channels[3], decoder_channels[4], skip_channels=64)
        
        self.out_conv = nn.Conv2d(decoder_channels[4],var_num,3,padding=1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 首先对输入进行上采样
        if self.upsample_factor > 1:
            x = F.interpolate(x, size=(self.upsampled_height, self.upsampled_width), 
                            mode='bilinear', align_corners=True)
        
        # Encoder
        e0 = F.relu(self.encoder0_bn(self.encoder0(x)))
        e0_pool = self.encoder0_pool(e0)
        
        e1 = self.encoder1(e0_pool)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Decoder - 始终计算完整的decoder路径
        d4 = self.decoder4(e4)
        d3 = self.decoder3(F.interpolate(d4, size=e3.shape[2:]), e3)
        d2 = self.decoder2(F.interpolate(d3, size=e2.shape[2:]), e2)
        d1 = self.decoder1(F.interpolate(d2, size=e1.shape[2:]), e1)
        d0 = self.decoder0(F.interpolate(d1, size=e0.shape[2:]), e0)
        d0 = self.out_conv(d0)
        # 根据output_layer选择用于生成权重的特征图
        out = {
            0: d0,
            1: d1,
            2: d2,
            3: d3,
            4: d4,
            -1: e4,
        }[self.output_layer]

        # 生成权重
        # weights = self.weight_head(out)
        # batch_size = weights.shape[0]
        
        # 返回权重和最终的decoder输出
 
        return out, d0+x
        

class HyperNetFactory:
    _networks = {
        'resnet18unet': ResNet18UNetHyperNet
    }
    
    @classmethod
    def register_network(cls, name: str, network_class: type):
        cls._networks[name] = network_class
    
    @classmethod
    def create_network(cls, name: str, **kwargs) -> BasicHyperNet:
        if name not in cls._networks:
            raise ValueError(f"Unknown network type: {name}")
        return cls._networks[name](**kwargs)
    
    @classmethod
    def get_available_networks(cls) -> list:
        return list(cls._networks.keys())

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, input_dim, d_model):
        super(PositionalEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, d_model)
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x shape: (batch_size, n, c)
        batch_size, n, _ = x.size()
        x = self.linear(x)
        # Create position indices for each n in the sequence
        positions = torch.arange(0, n, device=x.device).unsqueeze(0).expand(batch_size, n)
        
        # Get the positional embeddings for each position
        pos_embeddings = self.embedding(positions)
        
        return x + pos_embeddings

class ImplicitMLP(nn.Module):
    def __init__(self, 
                 input_weight_dim=512,
                 input_embed_dim=128,
                 coord_dim=2,
                 hidden_dim=64,
                 num_var=1,
                 ):
        super().__init__()
        
        self.num_var = num_var
        # 坐标编码
        self.pos_encoder = nn.Linear(coord_dim, input_embed_dim)
        # PositionalEmbedding(
        #                     max_len=100,
        #                     input_dim=coord_dim,
        #                     d_model=input_embed_dim)
        self.state_encoder = nn.Linear(num_var, input_embed_dim)
        # PositionalEmbedding(
        #                     max_len=100,
        #                     input_dim=num_var,
        #                     d_model=input_embed_dim)
        # 特征编码
        self.has_features = True
        if self.has_features:
            self.month_embedding = nn.Linear(1, input_embed_dim)
            self.day_embedding = nn.Linear(1, input_embed_dim)
            self.hour_embedding = nn.Linear(1, input_embed_dim)
            self.resolution_embedding = nn.Linear(1, input_embed_dim)
            self.dem_embedding = nn.Linear(1, input_embed_dim)
        
        self.input_weight_dim = input_weight_dim
        self.input_embed_dim = input_embed_dim
        # 计算MLP输入维度
        
        self.hidden_dim = hidden_dim
        
        self.weight_gen_1 = nn.Sequential(
                                nn.Conv1d(self.input_weight_dim, self.input_embed_dim, kernel_size=1),
                                nn.ReLU(),
                                nn.Linear(20*20, self.hidden_dim))
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        
        self.weight_gen_2 = nn.Sequential(
                                nn.Conv1d(self.input_weight_dim, self.hidden_dim, kernel_size=1),
                                nn.ReLU(),
                                nn.Linear(20*20, self.hidden_dim))
        self.ln2 = nn.LayerNorm(self.hidden_dim)
        
        self.fc1 = ResMLP(self.hidden_dim)
        self.fc2 = ResMLP(self.hidden_dim)
        
        self.out_fc = nn.Linear(self.hidden_dim, self.num_var)
                                
    def forward(self, coords, state, features=None, weights=None):
        
        # 编码坐标
        x = self.state_encoder(state)
        x = x + self.pos_encoder(coords)
        
        # 编码特征（如果有）
        if self.has_features and features is not None:
            x = x + self.month_embedding(features[:, :, 0:1]) + self.day_embedding(features[:, :, 1:2]) \
                  + self.hour_embedding(features[:, :, 2:3]) + self.resolution_embedding(features[:, :, 3:4]) \
                  + self.dem_embedding(features[:, :, 4:5])
        
        batch_size, dim, _, _ = weights.shape
        weights = weights.view(batch_size, dim, -1)
        
        feat = self.weight_gen_1(weights)
        
        x = torch.bmm(x, feat)
        x = self.ln1(x)
        x = nn.ReLU()(x)
        
        feat = self.weight_gen_2(weights)
        
        x = torch.bmm(x, feat)
        x = self.ln2(x)
        x = nn.ReLU()(x)
        
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.out_fc(x)
        
        return out
    
class CombinedModel(nn.Module):
    def __init__(self, 
                 hyper_network_type: str,
                 hyper_input_channels: int, 
                 hyper_height: int, 
                 hyper_width: int,
                 coord_dim: int = 2, 
                 hidden_dim: int = 64, 
                 input_weight_dim: int = 512,
                 input_embed_dim: int = 64,
                 hyper_kwargs: Dict = None):
        super().__init__()
        
        self.implicit_mlp = ImplicitMLP(
            coord_dim=coord_dim,
            hidden_dim=hidden_dim,
            input_weight_dim=input_weight_dim,
            input_embed_dim=input_embed_dim
            
        )
        
        kwargs = {
            'input_channels': hyper_input_channels,
            'height': hyper_height,
            'width': hyper_width,
        }
        if hyper_kwargs:
            kwargs.update(hyper_kwargs)
            
        self.hyper_net = HyperNetFactory.create_network(
            name=hyper_network_type,
            **kwargs
        )
        
    def forward(self, hyper_input, states, coords, features=None):
        
        weights, hyper_out = self.hyper_net(hyper_input)
        
        # 获取MLP的输出
        mlp_output = self.implicit_mlp(coords, states, features, weights)
        
        # 返回MLP输出和decoder特征
        return mlp_output, hyper_out
    
if __name__ == "__main__":
    config = {
        'hyper_network_type': 'resnet18unet',
        'hyper_input_channels': 1,
        'hyper_height': 320,
        'hyper_width': 320,
        'coord_dim': 2,
        'hidden_dim': 128,
        'input_weight_dim': 512,
        'input_embed_dim': 64,
        'hyper_kwargs': {
            'upsample_factor': 1,
            'output_layer': -1,
            'decoder_channels': [256, 128, 64, 32, 16]
        }
    }
    
    model = CombinedModel(**config)
    
    batch_size = 4
    hyper_input = torch.randn(batch_size, config['hyper_input_channels'], 
                             config['hyper_height'], config['hyper_width'])
    coords = torch.randn(batch_size, 100, config['coord_dim'])
    features = torch.randn(batch_size, 100, 5)
    
    output,out = model(hyper_input, torch.randn(batch_size, 100, 1), coords, features)
    import pdb
        
    pdb.set_trace()
    print(f"Output shape: {output.shape}")