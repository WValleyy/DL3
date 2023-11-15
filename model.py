import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class Middle(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlock(nn.Module):

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None):
        super().__init__()

        if up_conv_in_channels is None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels is None:
            up_conv_out_channels = out_channels

      
        self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class Res50Unet(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=2):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        encode_blocks = []
        decode_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]

        self.input_pool = list(resnet.children())[3]

        # Encoder Blocks
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                encode_blocks.append(bottleneck)
        self.encode_blocks = nn.ModuleList(encode_blocks)

        #BottleNeck
        self.middle = Middle(2048, 2048)

        #Decoder Blocks
        decode_blocks.append(UpBlock(2048, 1024))
        decode_blocks.append(UpBlock(1024, 512))
        decode_blocks.append(UpBlock(512, 256))
        decode_blocks.append(UpBlock(in_channels=128 + 64, out_channels=128, up_conv_in_channels=256, up_conv_out_channels=128))
        decode_blocks.append(UpBlock(in_channels=64 + 3, out_channels=64, up_conv_in_channels=128, up_conv_out_channels=64))

        self.decode_blocks = nn.ModuleList(decode_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools["layer_0"] = x
        x = self.input_block(x)
        pre_pools["layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.encode_blocks, 2):
            x = block(x)
            if i == (Res50Unet.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.middle(x)

        for i, block in enumerate(self.decode_blocks, 1):
            key = f"layer_{Res50Unet.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        x = self.out(x)
        del pre_pools
        
        return x
