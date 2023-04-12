# This file is mostly taken from BTS; author: Jin Han Lee, with only slight modifications
import os
import random

import json
import h5py
import numpy as np
from copy import deepcopy
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from ..utils.dataloader import seed_worker, sample_point_from_hist_parallel, get_hist_parallel, patch_info_from_rect_data

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class NYUV2(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))

            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   worker_init_fn=seed_worker)

        elif mode == 'online_eval':
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
        fname = args.filenames_file
        if mode == 'online_eval':
            md = 'test'
        else:
            md = 'train'

        with open(fname, 'r') as json_file:
            print(fname)
            json_data = json.load(json_file)
            self.sample_list = json_data[md]

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval

        self.K_list = torch.Tensor([
            5.1885790117450188e+02,
            5.1946961112127485e+02,
            3.2558244941119034e+02 - 16.0,
            2.5373616633400465e+02 - 12.0
        ])

    def __getitem__(self, idx):
        focal = self.K_list[0].item()

        path_file = os.path.join(self.args.data_path,
                                self.sample_list[idx]['filename'])

        f = h5py.File(path_file, 'r')
        rgb_h5 = f['rgb'][:].transpose(1, 2, 0)
        dep_h5 = f['depth'][:]

        image = Image.fromarray(rgb_h5, mode='RGB')
        depth_gt = Image.fromarray(dep_h5.astype('float32'), mode='F')

        if self.mode == 'train':
            # To avoid blank boundaries due to pixel registration
            depth_gt = depth_gt.crop((16, 12, 640-16, 480-12))
            image = image.crop((16, 12, 640-16, 480-12))

            if self.args.do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

            image = np.array(image)
            depth_gt = np.array(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            image, depth_gt = self.random_crop(image, depth_gt, self.args.input_height, self.args.input_width)
            image, depth_gt = self.train_preprocess(image, depth_gt)

            image = np.array(image, dtype=np.float32) / 255.0
            sample = {'image': image, 'depth': depth_gt, 'focal': focal}

        else:
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

        hist_data, fr, mask = get_hist_parallel(sample['image'], sample['depth'], self.args)
        if self.mode == 'train' and self.args.drop_hist > 1e-3:
            index = np.where(mask == True)[0]
            index = np.random.choice(index, int(len(index)*self.args.drop_hist))
            mask[index] = False
        if self.mode == 'train' and self.args.noise_prob > 1e-3:
            prob = np.random.random(hist_data[mask,0].shape)
            noise_mask = prob < self.args.noise_prob
            noise = np.random.normal(self.args.noise_mean, self.args.noise_sigma, hist_data[mask,0].shape)
            hist_data[mask,0][noise_mask] += noise[noise_mask]

        fh = sample_point_from_hist_parallel(hist_data, mask, self.args)
        patch_info = patch_info_from_rect_data(fr)
        sample['additional'] = {
            'hist_data': fh.to(torch.float),
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

