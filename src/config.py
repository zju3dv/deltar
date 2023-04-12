import argparse
import sys

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)

# Arguments
parser = argparse.ArgumentParser(description='Training script. Default values of all arguments are recommended for reproducibility', fromfile_prefix_chars='@',
                                conflict_handler='resolve')
parser.convert_arg_line_to_args = convert_arg_line_to_args
parser.add_argument('--epochs', default=25, type=int, help='number of total epochs to run')
parser.add_argument('--n-bins', '--n_bins', default=80, type=int,
                    help='number of bins/buckets to divide depth range into')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float, help='max learning rate')
parser.add_argument('--wd', '--weight-decay', default=0.1, type=float, help='weight decay')
parser.add_argument('--div-factor', '--div_factor', default=25, type=float, help="Initial div factor for lr")
parser.add_argument('--final-div-factor', '--final_div_factor', default=100, type=float,
                    help="final div factor for lr")
parser.add_argument('--bs', default=16, type=int, help='batch size')
parser.add_argument("--name", default="UnetAdaptiveBins")
parser.add_argument("--norm", default="linear", type=str, help="Type of norm/competition for bin-widths",
                    choices=['linear', 'softmax', 'sigmoid'])
parser.add_argument("--same-lr", '--same_lr', default=False, action="store_true",
                    help="Use same LR for all param groups")
parser.add_argument("--resume", default='', type=str, help="Resume from checkpoint")
parser.add_argument("--notes", default='', type=str, help="Wandb notes")
parser.add_argument("--tags", default='sweep', type=str, help="Wandb tags")
parser.add_argument("--workers", default=11, type=int, help="Number of workers for data loading")
parser.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")
parser.add_argument("--dataset_eval", default='realsense', type=str, help="Dataset to train on")
parser.add_argument("--data_path", default='../dataset/nyu/sync/', type=str,
                    help="path to dataset")
parser.add_argument('--filenames_file',
                    default="./train_test_inputs/nyudepthv2_train_files_with_gt.txt",
                    type=str, help='path to the filenames text file')
parser.add_argument('--data_path_eval',
                    default="../dataset/nyu/official_splits/test/",
                    type=str, help='path to the data for online evaluation')
parser.add_argument('--filenames_file_eval',
                    default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
                    type=str, help='path to the filenames text file for online evaluation')
parser.add_argument('--input_height', type=int, help='input height', default=416)
parser.add_argument('--input_width', type=int, help='input width', default=544)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)
parser.add_argument('--do_random_rotate', default=False,
                    help='if set, will perform random rotation for augmentation',
                    action='store_true')
parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10)
parser.add_argument('--no_logging', help='if set, will not logging', action='store_true')
parser.add_argument('--patch_size', help='conv patch size in miniViT', default=16, type=int)
parser.add_argument('--zone_sample_num', help='number of sampled points in each hist zone', default=16, type=int)
parser.add_argument('--save_for_demo', action='store_true')
parser.add_argument('--save_rgb', action='store_true')
parser.add_argument('--save_pred', action='store_true')
parser.add_argument('--save_error_map', action='store_true')
parser.add_argument('--save_dir', type=str, default='tmp')
parser.add_argument('--weight_path', help='')
parser.add_argument('--drop_hist', type=float, default=0.0)
parser.add_argument('--noise_mean', type=float, default=0.0)
parser.add_argument('--noise_sigma', type=float, default=0.0)
parser.add_argument('--noise_prob', type=float, default=0.0)
parser.add_argument('--train_zone_num', type=int, default=8)
parser.add_argument('--train_zone_random_offset', type=int, default=0)
parser.add_argument('--sample_uniform', action='store_true')
parser.add_argument('--attention_layer', default=['hist2image','image','hist2image','image'], nargs='+')
parser.add_argument('--validate-every', '--validate_every', default=100, type=int, help='validation period')
parser.add_argument('--simu_max_distance', type=float, default=4.0)

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

args.batch_size = args.bs
args.num_threads = args.workers
args.mode = 'train'
args.num_workers = args.workers
