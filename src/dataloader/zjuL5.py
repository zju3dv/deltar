# This file is mostly taken from BTS; author: Jin Han Lee, with only slight modifications
import os
import random
import json
import h5py
import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from copy import deepcopy
from ..utils.dataloader import seed_worker, sample_point_from_hist_parallel, get_hist_parallel, patch_info_from_rect_data

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class ZJUL5(object):
    def __init__(self, args, mode):
        assert mode == 'online_eval'
        self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
        self.data = DataLoader(self.testing_samples, 1,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=False)

class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = deepcopy(args)
        self.args.mode = mode

        fname, md = None, None
        if mode == 'online_eval':
            md = 'test'
            self.fname_json = args.filenames_file_eval
            self.data_path = args.data_path_eval

        with open(self.fname_json, 'r') as json_file:
            print(self.fname_json)
            json_data = json.load(json_file)
            self.sample_list = json_data[md]

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval

        self.K_list = torch.Tensor([
            611.2,
            609.6,
            323.4,
            244.9
        ])
        self.zone_num = 8

    def __getitem__(self, idx):
        # import ipdb; ipdb.set_trace()
        focal = self.K_list[0].item()

        path_file = os.path.join(self.data_path,
                                self.sample_list[idx]['filename'])

        f = h5py.File(path_file, 'r')
        rgb_h5 = f['rgb'][:]
        dep_h5 = f['depth'][:]

        image = Image.fromarray(rgb_h5, mode='RGB')
        depth_gt = Image.fromarray(dep_h5.astype('float32'), mode='F')

        image = np.array(image, dtype=np.float32) / 255.0
        depth_gt = np.array(depth_gt, dtype=np.float32)
        depth_gt = np.expand_dims(depth_gt, axis=2)
        fname = self.sample_list[idx]['filename']
        image_path = fname[fname.rfind('/')+1:].replace('h5', 'jpg')
        image_folder = fname[:fname.rfind('/')]

        if self.mode == 'online_eval':
            sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': True,
                        'image_path': image_path, 'image_folder': image_folder}
        else:
            sample = {'image': image, 'focal': focal}

        if self.transform:
            sample = self.transform(sample)

        hist_data, fr, mask = torch.from_numpy(f['hist_data'][:]), torch.from_numpy(f['fr'][:]), torch.from_numpy(f['mask'][:])

        fh = sample_point_from_hist_parallel(hist_data, mask, self.args)
        patch_info = patch_info_from_rect_data(fr)
        sample['additional'] = {
            'hist_data': fh.to(torch.float),
            'raw_data': hist_data.to(torch.float),
            'rect_data': fr.to(torch.float),
            'mask': mask,
            'patch_info': patch_info
        }

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        brightness = random.uniform(0.75, 1.25)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.sample_list)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        # import ipdb; ipdb.set_trace()
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth, 'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth,
                    'image_path': sample['image_path'], 'image_folder': sample['image_folder']}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            # img = torch.from_numpy(pic.copy().transpose((2, 0, 1)))
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
