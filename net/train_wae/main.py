"""main.py"""
import os
import sys
sys.path.append(os.path.abspath('../../../'))
from net.train.train_wae import trainer_mmd # , trainer_gan
from net.train.train_wae.utils import str2bool

import argparse
import numpy as np
import torch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    Trainer = trainer_mmd.Trainer

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='toy Beta-VAE')

    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--cuda', default=False, type=str2bool, help='enable cuda')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device')
    parser.add_argument('--max_epoch', default=90, type=float, help='maximum training epoch')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--workers', default=6, type=int, help='workers')
    parser.add_argument('--img_size', default=64, type=int, help='image size')

    parser.add_argument('--model', default='mmd', type=str, help='model for wae. mmd/gan')
    parser.add_argument('--z_dim', default=2048, type=int, help='dimension of the latent z')
    parser.add_argument('--z_var', default=2, type=int, help='scalar variance of the isotropic gaussian prior P(Z)')
    parser.add_argument('--reg_weight', default=100, type=float, help='lambda parameter for the regularization term')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate for AE')
    parser.add_argument('--lr_D', default=1e-4, type=float, help='learning rate for Adversary Network')
    parser.add_argument('--beta1', default=0.5, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')

    parser.add_argument('--dset_dir', default='/media/x/工作/datasets/ILSVRC2015/Data/VID', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset name')

    parser.add_argument('--viz_on', default=True, type=str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_name', default='main_2048', type=str, help='visdom env name')
    parser.add_argument('--viz_port', default=8097, type=str, help='visdom port number')
    parser.add_argument('--save_output', default=True, type=str2bool, help='save traverse images and gif')
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')

    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='last', type=str, help='load previous checkpoint. insert checkpoint filename')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
