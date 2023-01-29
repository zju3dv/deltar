import torch
import torch.nn as nn
from .fusion import TransformerFusion
import torch.nn.functional as F


class DepthRegression(nn.Module):
    def __init__(self, in_channels, dim_out=256, embedding_dim=128, norm='linear'):
        super(DepthRegression, self).__init__()
        self.norm = norm

        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))

    def forward(self, x):
        range_attention_maps = self.conv3x3(x)
        regression_head = self.conv1x1(x)
        regression_head = regression_head.mean([2,3])

        y = self.regressor(regression_head)  # .shape = N, dim_out
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)
        return y, range_attention_maps


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        if concat_with is None:
            up_x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            f = up_x
        else:
            up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
            f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class Decoder(nn.Module):
    def __init__(self, num_classes=1):
        super(Decoder, self).__init__()

        # [1/16, 1/8, 1/4, 1/2]
        # [136, 56, 40, 16]
        encoder_channels = [232, 136, 56, 40, 16]
        decoder_channels = [256, 256, 128, 64, 32]

        self.conv4 = nn.Conv2d(encoder_channels[0], decoder_channels[0], kernel_size=1, stride=1, padding=0)
        self.up1 = UpSampleBN(skip_input=decoder_channels[0] + encoder_channels[1], output_features=decoder_channels[1])
        self.up2 = UpSampleBN(skip_input=decoder_channels[1] + encoder_channels[2], output_features=decoder_channels[2])
        self.up3 = UpSampleBN(skip_input=decoder_channels[2] + encoder_channels[3], output_features=decoder_channels[3])
        self.up4 = UpSampleBN(skip_input=decoder_channels[3] + encoder_channels[4], output_features=decoder_channels[4])

        self.conv3 = nn.Conv2d(decoder_channels[1], decoder_channels[2], kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(decoder_channels[2], decoder_channels[3], kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(decoder_channels[3], decoder_channels[4], kernel_size=1, stride=1, padding=0)

        self.conv0 = nn.Conv2d(decoder_channels[4], num_classes, kernel_size=3, stride=1, padding=1)

        resolution = [
            [240, 320],
            [120, 160],
            [60, 80],
            [30, 40],
            [15, 20],
        ]

        channels = [int(c/2) for c in decoder_channels]

        self.cross_atten1 = TransformerFusion(embedding_dim=channels[3], max_resolution=resolution[1])
        self.cross_atten2 = TransformerFusion(embedding_dim=channels[2], max_resolution=resolution[2])
        self.cross_atten3 = TransformerFusion(embedding_dim=channels[1], max_resolution=resolution[3])

    def forward(self, img_features, hist_features, **kwargs):
        x_block0, x_block1, x_block2, x_block3, x_block4 = img_features
        depth_feat1, depth_feat2, depth_feat3 = hist_features

        # x_block0 -> (b, 16, h/2, w/2)
        # x_block1 -> (b, 40, h/4, w/4)
        # x_block2 -> (b, 56, h/8, w/8)
        # x_block3 -> (b, 136, h/16, w/16)
        # x_block4 -> (b, 232, h/32, w/32)
        # x_d4 = self.conv4(x_block4)

        x_d4 = self.conv4(x_block4)

        x_d3 = self.up1(x_d4, x_block3)
        x_d3 = self.conv3(x_d3)
        x_d3_fused = self.cross_atten3(x_d3, depth_feat3, **kwargs)
        x_d3 = torch.cat([x_d3, x_d3_fused], dim=1)

        x_d2 = self.up2(x_d3, x_block2)
        x_d2 = self.conv2(x_d2)
        x_d2_fused = self.cross_atten2(x_d2, depth_feat2, **kwargs)
        x_d2 = torch.cat([x_d2, x_d2_fused], dim=1)

        x_d1 = self.up3(x_d2, x_block1)
        x_d1 = self.conv1(x_d1)
        x_d1_fused = self.cross_atten1(x_d1, depth_feat1, **kwargs)
        x_d1 = torch.cat([x_d1, x_d1_fused], dim=1)

        x_d0 = self.up4(x_d1, x_block0)

        out = self.conv0(x_d0)

        return out

