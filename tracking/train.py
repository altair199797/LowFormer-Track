import os
import argparse
import random
import sys
sys.path.append('.')
from lib.train.run_training import run_training

def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for MobileViT training')

    parser.add_argument('--script', type=str, default='mobilevitv2_track', help='training script name')
    parser.add_argument('--config', type=str, default='mobilevitv2_256_128x1_ep300', help='yaml configure file')

    parser.add_argument('--save_dir', type=str, default='./output/', help='root directory to save checkpoints, logs, and tensorboard')
    parser.add_argument('--mode', type=str, choices=["single", "multiple", "multi_node"], default="single", help="train on single gpu or multiple gpus")
    parser.add_argument('--nproc_per_node', type=int, default=1, help="number of GPUs per node")  # specify when mode is multiple
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)  # whether datasets are in lmdb format
    parser.add_argument('--script_prv', type=str, help='training script name')
    parser.add_argument('--config_prv', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--use_wandb', type=int, choices=[0, 1], default=0)  # whether to use wandb

    # for knowledge distillation
    parser.add_argument('--distill', type=int, choices=[0, 1], default=0)  # whether to use knowledge distillation
    parser.add_argument('--script_teacher', type=str, help='teacher script name')
    parser.add_argument('--config_teacher', type=str, help='teacher yaml configure file name')

    # for multiple machines
    parser.add_argument('--rank', type=int, help='Rank of the current process.')
    parser.add_argument('--world-size', type=int, help='Number of processes participating in the job.')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP of the current rank 0.')
    parser.add_argument('--port', type=int, default='20000', help='Port of the current rank 0.')

    args = parser.parse_args()

    return args

# python tracking/train.py --script mobilevitv2_track --config mobilevitv2_256_128x1_ep300_coco --save_dir ./output --mode single
# python tracking/train.py --script lowformer_track --config lowformer_256_128x1_ep300_lasot_b15 --save_dir ./output --mode single

### MULTIPLE
# CUDA_VISIBLE_DEVICES=1,2,3,4,5 python tracking/train.py --script lowformer_track --config lowformer_256_128x1_ep300_all_b15_lff --save_dir ./output --mode multiple --nproc_per_node 5
# python tracking/train.py --script lowformer_track --config lowformer_256_128x1_ep300_lasot_coco_got_b15_lffv3_stridehead --save_dir ./output --mode single


# python tracking/train.py --script mobilevitv2_track --config mobilevitv2_256_128x1_ep300_lasot_got_coco --save_dir ./output --mode single

def main():
    args = parse_args()
    if args.mode == "single":
        """
        train_cmd = "python lib/train/run_training.py --script %s --config %s --save_dir %s --use_lmdb %d " \
                    "--script_prv %s --config_prv %s --distill %d --script_teacher %s --config_teacher %s --use_wandb %d"\
                    % (args.script, args.config, args.save_dir, args.use_lmdb, args.script_prv, args.config_prv,
                       args.distill, args.script_teacher, args.config_teacher, args.use_wandb)
        """
        run_training(script_name = args.script, config_name = args.config, save_dir = args.save_dir)

    elif args.mode == "multiple":
        train_cmd = "python -m torch.distributed.launch --nproc_per_node %d --master_port %d lib/train/run_training.py " \
                    "--script %s --config %s --save_dir %s --use_lmdb %d --script_prv %s --config_prv %s " \
                    "--use_wandb %d --distill %d --script_teacher %s --config_teacher %s" \
                    % (args.nproc_per_node, random.randint(10000, 50000), args.script, args.config, args.save_dir, args.use_lmdb,
                       args.script_prv, args.config_prv, args.use_wandb, args.distill, args.script_teacher, args.config_teacher)
        os.system(train_cmd)
    else:
        raise ValueError("mode should be 'single' or 'multiple'.")


if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=6,7 python tracking/train.py --script mobilevitv2_track --config mobilevitv2_256_128x1_ep300_lowformit --save_dir ./output --mode multiple --nproc_per_node 2