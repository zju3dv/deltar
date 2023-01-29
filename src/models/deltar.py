import torch
import torch.nn as nn

from .decoder import DepthRegression
from .encoder import ImageEncoder, HistogramEncoder
from .decoder import Decoder

class Deltar(nn.Module):
    def __init__(self, n_bins=100, min_val=0.1, max_val=10, norm='linear'):
        super(Deltar, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.img_encoder = ImageEncoder()
        self.hist_encoder = HistogramEncoder()
        self.depth_head = DepthRegression(128, dim_out=n_bins, norm=norm)
        self.decoder = Decoder(num_classes=128)
        self.conv_out = nn.Sequential(nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))

        self._reset_parameters()

    def _reset_parameters(self):
        # import ipdb; ipdb.set_trace()
        modules =[self.depth_head, self.decoder, self.conv_out] 
        for s in modules:
            for m in s.modules():
                if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_data, **kwargs):
        x = input_data['rgb']

        additional_data = input_data['additional']

        img_features = self.img_encoder(x)
        hist_features = self.hist_encoder(additional_data['hist_data'].unsqueeze(-1))
        kwargs.update({
            'rect_data': additional_data['rect_data'],
            'mask': additional_data['mask'],
            'patch_info': additional_data['patch_info'],
        })

        unet_out = self.decoder(img_features, hist_features, **kwargs)

        bin_widths_normed, range_attention_maps = self.depth_head(unet_out)
        out = self.conv_out(range_attention_maps)

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)

        return bin_edges, pred

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.img_encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.depth_head, self.conv_out]
        for m in modules:
            yield from m.parameters()
        

if __name__ == '__main__':
    model = Deltar.build(100)
    x = torch.rand(2, 3, 480, 640)
    bins, pred = model(x)
    print(bins.shape, pred.shape)
