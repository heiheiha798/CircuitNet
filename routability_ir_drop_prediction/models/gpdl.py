# Copyright 2022 CircuitNet. All rights reserved.
# Model modified to include SE Blocks and ASPP by default.

import torch
import torch.nn as nn
import torch.nn.functional as F 
from collections import OrderedDict

# 权重初始化和加载函数 (保持不变)
def generation_init_weights(module):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    module.apply(init_func)

def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')
    load(module)
    load = None
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]
    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')
    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    return missing_keys

# --- Squeeze-and-Excitation Block ---
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --- conv 模块修改: 加入 SEBlock ---
class conv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, use_se=True): # 默认启用SE
        super(conv, self).__init__()
        self.main_conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(dim_out)

    def forward(self, input):
        x = self.main_conv(input)
        if self.use_se:
            x = self.se(x)
        return x

# --- upconv 模块 (保持不变) ---
class upconv(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(upconv, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)

# --- Encoder 模块 (参数化输出通道，其他与原版相似) ---
class Encoder(nn.Module):
    def __init__(self, in_dim=3, c1_channels=32, c2_channels=64, bottleneck_channels=32, use_se_in_conv=True):
        super(Encoder, self).__init__()
        self.in_dim = in_dim
        self.c1 = conv(in_dim, c1_channels, use_se=use_se_in_conv)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c2 = conv(c1_channels, c2_channels, use_se=use_se_in_conv)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3_bottleneck = nn.Sequential( # 重命名c3以示其为瓶颈层
            nn.Conv2d(c2_channels, bottleneck_channels, 3, 1, 1),
            nn.BatchNorm2d(bottleneck_channels), # 原版此处用BN
            nn.Tanh()
        )
        # 用于skip connection的h2的通道数
        self.skip_channels_h2 = c1_channels 

    def init_weights(self):
        generation_init_weights(self)
        
    def forward(self, input):
        h1 = self.c1(input)            # 输出 c1_channels
        h2_for_skip = self.pool1(h1)   # 输出 c1_channels, 作为skip connection
        h3 = self.c2(h2_for_skip)      # 输出 c2_channels
        h4 = self.pool2(h3)
        h5_bottleneck = self.c3_bottleneck(h4) # 输出 bottleneck_channels
        return h5_bottleneck, h2_for_skip

# --- ASPP 模块 ---
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) # ASPP通常用ReLU
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)) # ASPP通常用ReLU

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=(6, 12, 18)):
        super(ASPP, self).__init__()
        # 每个ASPP分支输出的通道数
        # DeepLabV3+通常将每个分支输出设为256，然后project回一个较小值
        # 这里我们设为 in_channels // 4 或一个合理值，例如64
        # 如果in_channels本身就很小(如32)，则每个分支可以输出更小的通道数，例如in_channels
        # 我们让ASPP总输出为out_channels, 每个分支输出 out_channels // 5 (或调整)
        num_branches = len(atrous_rates) + 2 # 1x1, N atrous, 1 pooling
        if out_channels % num_branches == 0:
             branch_channels = out_channels // num_branches
        else:
            # 如果不能整除，可以考虑固定branch_channels，然后调整project层的输入
            # 或者让project层输出固定的out_channels
            branch_channels = max(32, out_channels // 4) # 确保至少有一定通道数

        self.convs = nn.ModuleList()
        # 1x1 convolution
        self.convs.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)))
        # Atrous convolutions
        for rate in atrous_rates:
            self.convs.append(ASPPConv(in_channels, branch_channels, rate))
        # Image pooling
        self.convs.append(ASPPPooling(in_channels, branch_channels))

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * branch_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5) # 可选的Dropout
        )

    def forward(self, x):
        res = []
        for conv_branch in self.convs:
            res.append(conv_branch(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

# --- Decoder 模块 (适配ASPP输出和Encoder的skip connection) ---
class Decoder(nn.Module):
    def __init__(self, final_out_channels=2, # 最终输出的通道数
                 main_input_channels=32,   # 来自ASPP的输入通道数
                 skip_input_channels=32,   # 来自Encoder的skip connection通道数
                 use_se_in_conv=True):
        super(Decoder, self).__init__()
        # conv1处理来自ASPP的特征
        self.conv1 = conv(main_input_channels, 32, use_se=use_se_in_conv) # 输出32通道
        self.upc1 = upconv(32, 16) # 输出16通道，尺寸放大 (H/2, W/2)
        
        # conv2的输入是 upc1的输出 (16) concat skip_input_channels (来自Encoder的h2_for_skip, 32通道)
        # 所以 conv2 的输入通道数是 16 + skip_input_channels
        self.conv2 = conv(16 + skip_input_channels, 16, use_se=use_se_in_conv) # 输出16通道
        
        # upc2的输入是 conv2的输出 (16通道)
        # 注意：原版GPDL的upc2输入是 torch.cat([d3 (16), skip (32)], dim=1) = 48
        # 如果这里的conv2输出16, 它会和更早的skip (h1) concat.
        # 为了保持结构简单，我们让upc2直接处理conv2的输出
        # U-Net的第二个上采样后，通常会concat Encoder第一层 (或接近第一层)的skip
        # 此处我们简化，暂时不加入第二个skip connection，如果需要，可以再添加
        # 根据您原始Decoder, upc2输入是 torch.cat([d3, skip], dim=1)
        # d3是conv2的输出(16), skip是来自Encoder的h2 (32). 所以是48 -> upc2(48,4)
        # 我们这里conv2输出16，需要修改upc2的输入通道
        # 我们在Encoder中选择h2_for_skip作为第一个也是唯一的skip
        # 因此，upc1的输出 d2(16) 与 h2_for_skip(32) 拼接后输入给conv2，conv2输入(16+32)=48
        # self.conv2 = conv(16 + skip_input_channels, 16, use_se=use_se_in_conv)
        # 然后upc2的输入是conv2的输出，即16
        self.upc2 = upconv(16, 4) # 输出4通道，尺寸放大 (H, W)
        
        self.conv3_final_pred = nn.Sequential( # 重命名conv3以示其为最终预测层
            nn.Conv2d(4, final_out_channels, 3, 1, 1),
            nn.Sigmoid()
        )

    def init_weights(self):
        generation_init_weights(self)

    def forward(self, main_feature, skip_feature): # main_feature来自ASPP, skip_feature来自Encoder的h2_for_skip
        d1 = self.conv1(main_feature) # 32通道, H/4, W/4 (假设main_feature是这个尺寸)
        d2_upsampled = self.upc1(d1)  # 16通道, H/2, W/2
        
        # Concat d2_upsampled (16 channels) with skip_feature (e.g., 32 channels from Encoder h2_for_skip)
        d_concat = torch.cat([d2_upsampled, skip_feature], dim=1) # 16 + 32 = 48 通道

        # conv2 处理 concat后的特征
        # self.conv2输入通道需要是48
        # 为了保持conv2的输入为16 + skip_input_channels (即48)，
        # 我们将 self.conv2 的 dim_in 改为动态传入或直接设为48
        # 或者，保持conv2为 (16+skip_input_channels, 16), 即 (48, 16)
        # 在__init__中，conv2的dim_in已经设为 16 + skip_input_channels
        d3_processed = self.conv2(d_concat) # 输出16通道
        
        d4_upsampled = self.upc2(d3_processed) # 4通道, H, W
        output = self.conv3_final_pred(d4_upsampled)
        return output

# --- GPDL 主模型 (集成Encoder, ASPP, Decoder) ---
class GPDL(nn.Module):
    def __init__(self,
                 in_channels=3,          # 输入图像通道数
                 out_channels=2,         # 最终分割输出的类别数
                 # Encoder参数
                 enc_c1_channels=32,
                 enc_c2_channels=64,
                 enc_bottleneck_channels=32, # Encoder瓶颈输出, 作为ASPP输入
                 # ASPP参数
                 aspp_out_channels=32,       # ASPP输出通道数, 作为Decoder主要输入
                 aspp_atrous_rates=(6, 12, 18),
                 # Decoder参数 (skip_input_channels由Encoder决定，这里是enc_c1_channels)
                 # SE Block控制
                 use_se_in_conv=True,
                 **kwargs): # 兼容旧的configs
        super().__init__()

        self.encoder = Encoder(in_dim=in_channels,
                               c1_channels=enc_c1_channels,
                               c2_channels=enc_c2_channels,
                               bottleneck_channels=enc_bottleneck_channels,
                               use_se_in_conv=use_se_in_conv)
        
        self.aspp = ASPP(in_channels=enc_bottleneck_channels,
                         out_channels=aspp_out_channels,
                         atrous_rates=aspp_atrous_rates)
        
        # Decoder的skip_input_channels来自Encoder中h2_for_skip的通道数 (即enc_c1_channels)
        self.decoder = Decoder(final_out_channels=out_channels,
                               main_input_channels=aspp_out_channels,
                               skip_input_channels=self.encoder.skip_channels_h2, # 直接从encoder实例获取
                               use_se_in_conv=use_se_in_conv)

    def forward(self, x):
        bottleneck_feature, skip_feature = self.encoder(x)
        aspp_out = self.aspp(bottleneck_feature)
        output = self.decoder(aspp_out, skip_feature)
        return output

    def init_weights(self, pretrained=None, strict=False, **kwargs): # 保持与原版一致
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location='cpu')['state_dict']
            for k in weight.keys():
                new_dict[k] = weight[k]
            load_state_dict(self, new_dict, strict=strict, logger=None)
        elif pretrained is None:
            generation_init_weights(self)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')