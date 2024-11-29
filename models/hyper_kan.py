import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from typing import List, Optional, Dict, Tuple
import os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
torch.set_num_threads(1)

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()
        

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


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

class ImplicitKAN(nn.Module):
    def __init__(self, 
                layers_hidden,
                input_embed_dim=64,
                hidden_dim=64,
                grid_size=5,
                spline_order=3,
                scale_noise=0.1,
                scale_base=1.0,
                scale_spline=1.0,
                base_activation=torch.nn.SiLU,
                grid_eps=0.02,
                grid_range=[-1, 1],
                 input_weight_dim=512,
                 coord_dim=2,
                 num_var=1,
                 ):
        super().__init__()
        
        self.num_var = num_var
        # 坐标编码
        
        self.pos_encoder = nn.Linear(coord_dim, input_embed_dim)

        self.state_encoder = nn.Linear(num_var, input_embed_dim)

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
        
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
        
        self.fc_neck = nn.Linear(hidden_dim, layers_hidden[0])
        self.out_fc = nn.Linear(layers_hidden[-1], 1)
                                
    def forward(self, coords, state, features=None, weights=None, update_grid=False):
        
        # 编码坐标
        x = self.state_encoder(state)
        import pdb
        # pdb.set_trace()
        x = x + self.pos_encoder(coords)
        
        # 编码特征（如果有）
        if features is not None:
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
        import pdb
        # pdb.set_trace()
        x = self.fc_neck(x)
        
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
            
        out = self.out_fc(x)
        
        return out
    
class CombinedModelKAN(nn.Module):
    def __init__(self, 
                 hyper_network_type: str,
                 hyper_input_channels: int, 
                 hyper_height: int, 
                 hyper_width: int,
                 coord_dim: int = 2, 
                 hidden_dim: int = 64, 
                 input_weight_dim: int = 512,
                 input_embed_dim: int = 64,
                 hyper_kwargs: Dict = None,
                 layers_hidden = [64,64]):
        super().__init__()
        
        self.implicit_kan = ImplicitKAN(
            layers_hidden=layers_hidden,
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
        mlp_output = self.implicit_kan(coords, states, features, weights)
        
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