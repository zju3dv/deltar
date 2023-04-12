import torch
import random
import numpy as np
import math
import torch.nn.functional as F
from einops import rearrange, repeat


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

def patch_info_from_rect_data(rect_data):
    # extract patch width/height from rect_data
    # assume max_patch_size across batch are the same
    ret = {}
    zone_num = int(math.sqrt(rect_data.shape[0]))
    max_patch_height = torch.max(rect_data[...,2] - rect_data[...,0]).to(torch.int32).item()
    max_patch_width = torch.max(rect_data[...,3] - rect_data[...,1]).to(torch.int32).item()
    _pad_height = int(max(torch.max(torch.abs(torch.clip(rect_data[...,0],None,0))).item(),
                        torch.max(torch.clip(rect_data[...,2],480,None)-480).item()))
    _pad_width = int(max(torch.max(torch.abs(torch.clip(rect_data[...,1],None,0))).item(),
                        torch.max(torch.clip(rect_data[...,3],640,None)-640).item()))
    for conv_patch_size in [4, 8, 16]:
        pad_height = math.ceil(_pad_height/conv_patch_size)
        pad_width = math.ceil(_pad_width/conv_patch_size)
        sy_wo_pad = torch.min(rect_data[...,0]/conv_patch_size).to(torch.int32)
        sx_wo_pad = torch.min(rect_data[...,1]/conv_patch_size).to(torch.int32)
        p1 = math.ceil(max_patch_height/conv_patch_size)
        p2 = math.ceil(max_patch_width/conv_patch_size)
        ey_wo_pad = torch.max(rect_data[...,2]/conv_patch_size).to(torch.int32)
        ex_wo_pad = torch.max(rect_data[...,3]/conv_patch_size).to(torch.int32)
        ret[conv_patch_size] = {
            'pad_size': torch.tensor([pad_height, pad_width], dtype=torch.int),
            'patch_size': torch.tensor([p1, p2], dtype=torch.int),
            'index_wo_pad': torch.tensor([sy_wo_pad, sx_wo_pad, ey_wo_pad, ex_wo_pad], dtype=torch.int),
        }
    ret['zone_num'] = zone_num

    return ret


def tensor_linspace(start, end, steps=10):
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


def line(x, x1, y1, x2, y2):
    return (x-x1)/(x2-x1)*(y2-y1)+y1


def sample_point_from_hist_parallel(hist_data, mask, config):
    znum = int(math.sqrt(mask.numel()))
    fh = torch.zeros([znum**2, config.zone_sample_num], dtype=torch.float32)
    zone_sample_num = config.zone_sample_num
    if not config.sample_uniform:
        delta = 1e-3
        sample_ppf = torch.Tensor(np.arange(delta, 1, (1-2*delta)/(zone_sample_num-1)).tolist()).unsqueeze(0)
        d = torch.distributions.Normal(hist_data[mask, 0:1], hist_data[mask, 1:2])
        fh[mask] = d.icdf(sample_ppf).to(torch.float32)
    else:
        sigma = hist_data[mask, 1]
        start = hist_data[mask, 0] - 3.0*sigma
        end = hist_data[mask, 0] + 3.0*sigma
        depth = tensor_linspace(start, end, steps=config.zone_sample_num)
        fh[mask] = depth.to(torch.float32)
    return fh


def get_hist_parallel(rgb, dep, config):
    # share same interval/area
    height, width = rgb.shape[1], rgb.shape[2]
    max_distance = config.simu_max_distance
    range_margin = list(np.arange(0, max_distance+1e-9, 0.04))
    if config.mode == 'train':
        patch_height, patch_width = 64, 64
    else:
        patch_height, patch_width = 56, 56
    offset = 0
    if config.train_zone_random_offset > 0:
        offset = random.randint(-config.train_zone_random_offset, config.train_zone_random_offset)
    train_zone_num = config.train_zone_num if config.mode == 'train' else 8
    sy = int((height - patch_height*train_zone_num) / 2) + offset
    sx = int((width - patch_width*train_zone_num) / 2) + offset
    dep_extracted = dep[:, sy:sy+patch_height*train_zone_num, sx:sx+patch_width*train_zone_num]
    dep_patches = dep_extracted.unfold(2, patch_width, patch_width).unfold(1, patch_height, patch_height)
    dep_patches = dep_patches.contiguous().view(-1, patch_height, patch_width)
    hist = torch.stack([torch.histc(x, bins=int(max_distance/0.04), min=0, max=max_distance) for x in dep_patches], 0)

    # choose cluster with strongest signal, then fit the distribution
    # first, mask out depth smaller than 4cm, which is usually invalid depth, i.e., zero
    hist[:,0] = 0
    hist = torch.clip(hist-20, 0, None)
    for i, bin_data in enumerate(hist):
        idx = np.where(bin_data!=0)[0]
        idx_split = np.split(idx,np.where(np.diff(idx)!=1)[0]+1)
        bin_data_split = np.split(bin_data[idx],np.where(np.diff(idx)!=1)[0]+1)
        signal = np.argmax([torch.sum(b) for b in bin_data_split])
        hist[i, :] = 0
        hist[i, idx_split[signal]] = bin_data_split[signal]

    dist = ((torch.Tensor(range_margin[1:]) + np.array(range_margin[:-1]))/2).unsqueeze(0)
    sy = torch.Tensor(list(range(sy, sy+patch_height*train_zone_num, patch_height)) * train_zone_num).view([train_zone_num, -1]).T.reshape([-1])
    sx = torch.Tensor(list(range(sx, sx+patch_width*train_zone_num, patch_width)) * train_zone_num)
    fr = torch.stack([sy, sx, sy+patch_height, sx+patch_width], dim=1)
        
    mask = torch.zeros([train_zone_num, train_zone_num], dtype=torch.bool)
    n = torch.sum(hist, dim=1)
    mask = n > 0
    mask = mask.reshape([-1])
    mu = torch.sum(dist * hist, dim=1) / (n+1e-9)
    std = torch.sqrt(torch.sum(hist * torch.pow(dist-mu.unsqueeze(-1), 2), dim=1)/(n+1e-9))+1e-9
    fh = torch.stack([mu, std], axis=1).reshape([train_zone_num,train_zone_num,2])
    fh = fh.reshape([-1, 2])

    return fh, fr, mask

