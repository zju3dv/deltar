import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from src.utils.model_io import load_weights
from src.dataloader.zjuL5 import ZJUL5
from src.models.deltar import Deltar
from src.utils.utils import RunningAverageDict, colorize
from src.config import args


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse,
                log_10=log_10, rmse_log=rmse_log, silog=silog, sq_rel=sq_rel)

def predict_tta(model, input_data, args):
    _, pred = model(input_data)
    pred = np.clip(pred.cpu().numpy(), args.min_depth, args.max_depth)
    pred = nn.functional.interpolate(torch.Tensor(pred), input_data['rgb'].shape[-2:], mode='bilinear', align_corners=True)

    return torch.Tensor(pred)

def eval(model, test_loader, args, device):
    if args.save_dir is not None and not os.path.exists(f'{args.save_dir}'):
        os.system(f'mkdir -p {args.save_dir}')

    metrics = RunningAverageDict()
    with torch.no_grad():
        model.eval()
        for index, batch in enumerate(tqdm(test_loader)):
            gt = batch['depth'].to(device)
            img = batch['image'].to(device)
            input_data = {'rgb': img}
            additional_data = {}
            additional_data['hist_data'] = batch['additional']['hist_data'].to(device)
            additional_data['rect_data'] = batch['additional']['rect_data'].to(device)
            additional_data['mask'] = batch['additional']['mask'].to(device)
            additional_data['patch_info'] = batch['additional']['patch_info']
            input_data.update({
                'additional': additional_data
            })

            final = predict_tta(model, input_data, args)
            final = final.squeeze().cpu().numpy()

            impath = f"{batch['image_path'][0].replace('/', '__').replace('.jpg', '')}"
            im_subfolder = batch['image_folder'][0]
            vis_folder = f'{im_subfolder}'
            if not os.path.exists(f'{args.save_dir}/{vis_folder}'):
                os.system(f'mkdir -p {args.save_dir}/{vis_folder}')

            if args.save_pred:
                pred_path = os.path.join(args.save_dir, vis_folder, f"{impath}_pred.png")
                pred = colorize(torch.from_numpy(final).unsqueeze(0), vmin=0.0, vmax=3.0, cmap='magma')
                Image.fromarray(pred).save(pred_path)

            if args.save_error_map:
                error_map = np.abs(gt.squeeze().cpu().numpy() - final)
                viz = colorize(torch.from_numpy(error_map).unsqueeze(0), vmin=0, vmax=1.2, cmap='jet')
                Image.fromarray(viz).save(os.path.join(args.save_dir, vis_folder, f"{impath}_error.png"))

            if args.save_rgb:
                # _mean = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                # _std = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                # img = img.cpu() * _std + _mean
                img = (img-torch.min(img))/(torch.max(img)-torch.min(img))
                img = img.squeeze().permute([1,2,0]).cpu().numpy()
                img = (img*255).astype(np.uint8)
                # import ipdb; ipdb.set_trace()
                rgb = img.copy()
                Image.fromarray(rgb).save(os.path.join(args.save_dir, vis_folder, f"{impath}_rgb.png"))

            if args.save_for_demo:
                pred = (final * 1000).astype('uint16')
                pred_path = os.path.join(args.save_dir, vis_folder, f"{impath}_demo.png")
                Image.fromarray(pred).save(pred_path)

            gt = gt.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt > args.min_depth, gt < args.max_depth)
            metrics.update(compute_errors(gt[valid_mask], final[valid_mask]))

    metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
    print(f"Metrics: {metrics}")


if __name__ == '__main__':

    device = torch.device('cuda:0')
    test_loader = ZJUL5(args, 'online_eval').data
    model = Deltar(n_bins=args.n_bins, min_val=args.min_depth,
                    max_val=args.max_depth, norm='linear').to(device)
    model = load_weights(model, args.weight_path)
    model = model.eval()

    eval(model, test_loader, args, device)
