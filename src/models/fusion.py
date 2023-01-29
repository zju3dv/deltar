import torch
import copy
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from ..config import args
from .transformer import LoFTREncoderLayer, TwinsTransformer


class TransformerFusion(nn.Module):
    def __init__(self, embedding_dim, max_resolution, num_heads=4):
        super(TransformerFusion, self).__init__()

        self.zone_sample_num = args.zone_sample_num
        self.max_resolution = max_resolution
        self.positional_encodings = nn.Parameter(torch.rand(max_resolution[0]*max_resolution[1], embedding_dim), requires_grad=True)
        self.positional_encodings2 = nn.Parameter(torch.rand(self.zone_sample_num, embedding_dim), requires_grad=True)

        # https://github.com/facebookresearch/dino/blob/main/vision_transformer.py#L161
        nn.init.trunc_normal_(self.positional_encodings, std=0.2)
        nn.init.trunc_normal_(self.positional_encodings2, std=0.2)

        self.layer_names = args.attention_layer
        encoder_layer = LoFTREncoderLayer(embedding_dim, num_heads)

        ws = math.ceil(math.sqrt(math.sqrt(self.max_resolution[0] * self.max_resolution[1])))
        image_encoder_layer = TwinsTransformer(embedding_dim, num_heads, ws=ws)
        layer_list = []
        for i in range(len(self.layer_names)):
            name = self.layer_names[i]
            if name == 'image': layer = copy.deepcopy(image_encoder_layer)
            else: layer = copy.deepcopy(encoder_layer)
            layer_list.append(layer)
        self.layers = nn.ModuleList(layer_list)

        self.conv_patch_size = 640 / self.max_resolution[1]

    def interpolate_pos_encoding_2d(self, pos, size):
        h, w = size
        pos_n = F.interpolate(pos.unsqueeze(0), size=[h, w], mode='bilinear', align_corners=True)[0]
        return pos_n

    def interpolate_pos_encoding_1d(self, pos, l):
        pos_n = F.interpolate(pos.unsqueeze(0), size=[l], mode='linear', align_corners=True)[0]
        return pos_n

    def forward(self, x, feat1, **kwargs):
    # def forward(self, x, feat1, feat1_mask, rect_data, patch_info):
        # import ipdb; ipdb.set_trace()

        # conv -> patch embeddings
        # zone_mask -> patches covered by 8x8 zones
        # hist_mask -> patches covered by non-zero 8x8 zones (those with dist in 0-4m)

        embeddings = x
        B, D, H, W = embeddings.size()
        zone_sample_num = feat1.size(2)
        # print(feat1_mask.flatten())

        # extract patch width/height from rect_data
        # assume max_patch_size across batch are the same
        rect_data = kwargs['rect_data']
        feat1_mask = kwargs['mask']
        patch_info = kwargs['patch_info']
        zone_num = patch_info['zone_num'][0]
        pad_size = patch_info[self.conv_patch_size]['pad_size']
        patch_size = patch_info[self.conv_patch_size]['patch_size']
        index_wo_pad = patch_info[self.conv_patch_size]['index_wo_pad']

        pad_height, pad_width = torch.max(pad_size, axis=0)[0]
        p1, p2 = torch.max(patch_size, axis=0)[0]
        sy_wo_pad, sx_wo_pad = torch.min(index_wo_pad, axis=0)[0][0:2]
        ey_wo_pad, ex_wo_pad = torch.max(index_wo_pad, axis=0)[0][2:4]
        sy, ey = sy_wo_pad+pad_height, ey_wo_pad+pad_height
        sx, ex = sx_wo_pad+pad_width, ex_wo_pad+pad_width
        tzh, tzw = ey-sy, ex-sx
        interpolate = False
        if (ey-sy) != p1*zone_num or (ex-sx) != p2*zone_num:
            interpolate = True

        # add positional encoding
        offset_y, offset_x = 0, 0
        if embeddings.shape[2] < self.max_resolution[0]:
            offset_y = torch.randint(0, self.max_resolution[0]-embeddings.shape[2]+1,[1])
        if embeddings.shape[3] < self.max_resolution[1]:
            offset_x = torch.randint(0, self.max_resolution[1]-embeddings.shape[3]+1,[1])
        positional_encodings = self.positional_encodings.view([self.max_resolution[0], self.max_resolution[1], -1])
        positional_encodings = positional_encodings[offset_y:offset_y+H,offset_x:offset_x+W,:]
        positional_encodings = positional_encodings.permute(2, 0, 1)
        embeddings = embeddings + positional_encodings
        feat0 = embeddings.flatten(2).permute(0, 2, 1)

        # prepare zone_mask and hist_mask
        # zone_mask -> b, (h, w), c
        # hist_mask -> (b zn zn) (p1 p2) c
        # firstly assign mask value False/True
        zone_mask = torch.zeros([B, H, W]).to(torch.bool).to(feat0.device)
        zone_mask[:, torch.clip(sy_wo_pad,0,H):torch.clip(ey_wo_pad,0,H), torch.clip(sx_wo_pad,0,W):torch.clip(ex_wo_pad,0,W)] = 1
        hist_mask = rearrange(feat1_mask, 'b (zn zn2) -> b zn zn2', zn=zone_num)
        hist_mask = repeat(hist_mask, 'b zn zn2 -> b (zn p1) (zn2 p2)', p1=p1, p2=p2)
        # secondly, reshape to corresponding shape
        zone_mask = rearrange(zone_mask, 'b h w -> b (h w)')
        zone_mask = repeat(zone_mask, 'b s -> b s c', c=D)
        hist_mask = rearrange(hist_mask, 'b (zn p1) (zn2 p2) -> (b zn zn2) (p1 p2)', zn=zone_num, p1=p1, p2=p2)
        hist_mask = repeat(hist_mask, 'bzz p1p2 -> bzz p1p2 c', c=D)
        if pad_height > 0 or pad_width > 0:
            pad_mask = torch.ones([B, D, tzh, tzw]).to(torch.bool).to(feat0.device)
            pad_mask[:,:,:torch.clip(0-sy_wo_pad,0,None)] = 0
            if torch.clip(ey_wo_pad-H,0,None) > 0: pad_mask[:,:,-torch.clip(ey_wo_pad-H,0,None):] = 0
            pad_mask[...,:torch.clip(0-sx_wo_pad,0,None)] = 0
            if torch.clip(ex_wo_pad-W,0,None) > 0: pad_mask[...,-torch.clip(ex_wo_pad-W,0,None):] = 0
            pad_mask = rearrange(pad_mask, 'b c tzh tzw -> (b tzh tzw c)')
        else:
            pad_mask = torch.ones([B*D*(ey-sy)*(ex-sx)]).to(torch.bool).to(feat0.device)

        # process hist feature
        positional_encodings2 = self.positional_encodings2
        feat1 = feat1 + positional_encodings2.unsqueeze(0).unsqueeze(0)
        feat1 = rearrange(feat1, 'b z n d -> (b z) n d')
        feat1_mask = rearrange(feat1_mask, 'b z -> (b z)')
        feat1_mask = repeat(feat1_mask, 'b -> b l', l=self.zone_sample_num)

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'image':
                feat0 = layer(feat0, (H, W))
            elif name == 'hist2image':
                # extract zone from full feature map
                feat0_unflatten = F.pad(embeddings, (pad_width, pad_width, pad_height, pad_height), 'constant', 0)
                zone_feature = feat0_unflatten[:, :, int(sy):int(ey), int(sx):int(ex)]
                # if need interpolate
                if interpolate:
                    zone_feature = F.interpolate(zone_feature, size=[zone_num*p1, zone_num*p2], mode='bilinear', align_corners=True)
                zone_feature = rearrange(zone_feature, 'b c (ph p1) (pw p2) -> (b ph pw) (p1 p2) c', p1=p1, p2=p2)
                zone_feature = layer(zone_feature, feat1)
                zone_feature[~hist_mask] = 0
                # interpolate back to original size
                if interpolate:
                    zone_feature = rearrange(zone_feature, '(b ph pw) (p1 p2) c -> b c (ph p1) (pw p2)', b=B, ph=zone_num, p1=p1)
                    zone_feature = F.interpolate(zone_feature, size=[tzh, tzw], mode='bilinear', align_corners=True)
                    zone_feature = rearrange(zone_feature, 'b c tzh tzw -> (b tzh tzw c)')
                else:
                    zone_feature = rearrange(zone_feature, '(b zn zn2) (p1 p2) c -> (b zn p1 zn2 p2 c)', 
                                                    zn=zone_num, zn2=zone_num, p1=p1, p2=p2)
                feat0[zone_mask] += zone_feature[pad_mask]

        feat0 = rearrange(feat0, 'b (h w) c -> b c h w', h=H)

        return feat0


