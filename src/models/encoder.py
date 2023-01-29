import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class PointNetEncoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PointNetEncoder, self).__init__()
        # self.feature_dim = [64, 128, 128]
        self.conv1 = torch.nn.Conv1d(in_channel, out_channel, 1)
        self.conv2 = torch.nn.Conv1d(out_channel, out_channel, 1)
        self.conv3 = torch.nn.Conv1d(out_channel, out_channel, 1)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.bn3 = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        B, N, D = x.size()
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x.permute(0, 2, 1)

class HistExtractor(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HistExtractor, self).__init__()
        self.pointnet_encoder = PointNetEncoder(in_channel, out_channel)

    def forward(self, hist_data):
        B, Z, N, D = hist_data.size()
        hist_feature = self.pointnet_encoder(hist_data.view([B*Z, N, D]))
        _, _, ND = hist_feature.size()
        return hist_feature.view([B, Z, N, ND])

class HistogramEncoder(nn.Module):
    def __init__(self):
        super(HistogramEncoder, self).__init__()
        channels = [32, 64, 128]
        self.hist_extractor1 = HistExtractor(in_channel=1, out_channel=channels[0])
        self.hist_extractor2 = HistExtractor(in_channel=channels[0], out_channel=channels[1])
        self.hist_extractor3 = HistExtractor(in_channel=channels[1], out_channel=channels[2])

    def forward(self, hist_data):
        depth_feat1 = self.hist_extractor1(hist_data)
        depth_feat2 = self.hist_extractor2(depth_feat1)
        depth_feat3 = self.hist_extractor3(depth_feat2)

        return [depth_feat1, depth_feat2, depth_feat3]



class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        net = timm.create_model('tf_efficientnetv2_b3', pretrained=True)
        self.conv0 = nn.Sequential(
            net._modules['conv_stem'],
            net._modules['bn1'],
            net._modules['blocks'][0],
        )
        self.conv1 = net._modules['blocks'][1]
        self.conv2 = net._modules['blocks'][2]
        self.conv3 = nn.Sequential(
            net._modules['blocks'][3],
            net._modules['blocks'][4]
        )
        self.conv4 = net._modules['blocks'][5]

    def forward(self, x):
        features = [x]
        features.append(self.conv0(features[-1]))
        features.append(self.conv1(features[-1]))
        features.append(self.conv2(features[-1]))
        features.append(self.conv3(features[-1]))
        features.append(self.conv4(features[-1]))

        return features[1:]
