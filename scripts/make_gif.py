import os
import numpy as np
import cv2
import tqdm
import h5py
import imageio
import argparse

def normalize(value, vmin=0.0, vmax=4.0):
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    value = (value*255.0).astype(np.uint8)
    return value


parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', required=True, default='demo/room')
parser.add_argument('--pred_folder', required=True, default='tmp/room')
args = parser.parse_args()

image_list = []
files = sorted(os.listdir(args.data_folder))
for f in tqdm.tqdm(files):
    if not f.endswith('h5'): continue
    fname = os.path.basename(f)
    h5_fname = f'{args.data_folder}/{f}'
    pred_fname = f'{args.pred_folder}/{os.path.splitext(f)[0]}_demo.png'

    h5_file = h5py.File(h5_fname, 'r')

    fr = h5_file['fr'][:]
    hist_data = h5_file['hist_data'][:]
    mask = h5_file['mask'][:]

    pad_size = int(5)
    vis_img = np.zeros([480, 320+640*2+pad_size*2, 3], dtype=np.uint8)
    L5_depth = np.zeros([480, 640])
    for i in range(mask.shape[0]):
        if not mask[i]: continue
        if fr[i,2] < 0 or fr[i,3] < 0: continue
        fr[i, 0] = np.clip(fr[i, 0], 0, 10000)
        fr[i, 1] = np.clip(fr[i, 1], 0, 10000)
        sy, sx, ey, ex = fr[i]
        L5_depth[sy:ey, sx:ex] = hist_data[i, 0]
    L5_depth = cv2.applyColorMap(normalize(L5_depth), cv2.COLORMAP_MAGMA)

    realsense_depth = h5_file['depth'][:]
    pred_depth = cv2.imread(pred_fname, -1).astype(np.float32) / 1000.0
    rgb = h5_file['rgb'][:]
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    realsense_depth = cv2.applyColorMap(normalize(realsense_depth), cv2.COLORMAP_MAGMA)
    pred_depth = cv2.applyColorMap(normalize(pred_depth), cv2.COLORMAP_MAGMA)

    rgb = cv2.resize(rgb, (320, 240-pad_size), interpolation = cv2.INTER_AREA)
    L5_depth = cv2.resize(L5_depth, (320, 240-pad_size), interpolation = cv2.INTER_AREA)

    vis_img[0:240-pad_size, 0:320, :] = rgb
    vis_img[-(240-pad_size)-1:-1, 0:320, :] = L5_depth
    vis_img[:, 320+pad_size:320+pad_size+640, :] = pred_depth
    vis_img[:, 320+640+pad_size*2:, :] = realsense_depth

    vis_img = cv2.resize(vis_img, (960, 300), interpolation = cv2.INTER_AREA)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    image_list.append(vis_img)

imageio.mimsave('demo.gif', image_list, fps=10)