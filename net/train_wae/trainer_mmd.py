"""trainer_mmd.py"""

import math
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import visdom
import torchvision.datasets as datasets
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
import torchvision.transforms as T
from net.train.train_wae.ops import cuda, multistep_lr_decay, mmd
from net.train.train_wae.utils import DataGather
from net.wae64 import WAE64


class Trainer(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_epoch = args.max_epoch
        self.global_epoch = 0
        self.global_iter = 0

        self.z_dim = args.z_dim
        self.z_var = args.z_var
        self.z_sigma = math.sqrt(args.z_var)
        self._lambda = args.reg_weight
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.lr_schedules = {30:2, 50:5, 100:10}
        self.decoder_dist = 'gaussian'

        net = WAE64
        # self.net = net(tr=True)
        self.net = net(tr=True).cuda()
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                    betas=(self.beta1, self.beta2))

        self.gather = DataGather()
        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on
        if self.viz_on:
            self.viz = visdom.Visdom(env=self.viz_name+'_lines', port=self.viz_port)
            self.win_recon = None
            self.win_mmd = None
            self.win_mu = None
            self.win_var = None

        self.ckpt_dir = Path(args.ckpt_dir).joinpath(args.viz_name)
        if not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        self.save_output = args.save_output
        self.output_dir = Path(args.output_dir).joinpath(args.viz_name)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dset_dir = args.dset_dir
        self.dataset = datasets.ImageFolder(
            os.path.join(self.dset_dir, 'train'),
            T.Compose([
            T.Resize((args.img_size, args.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ]))

        self.batch_size = args.batch_size
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None)


    def train(self):
        self.net.train()

        iters_per_epoch = len(self.data_loader)
        max_iter = self.max_epoch*iters_per_epoch
        pbar = tqdm(total=max_iter)
        with tqdm(total=max_iter) as pbar:
            pbar.update(self.global_iter)
            out = False
            while not out:
                for x in self.data_loader:
                    pbar.update(1)
                    self.global_iter += 1
                    if self.global_iter % iters_per_epoch == 0:
                        self.global_epoch += 1
                    self.optim = multistep_lr_decay(self.optim, self.global_epoch, self.lr_schedules)

                    x = Variable(x[0].cuda())
                    x_recon, z_tilde = self.net(x)
                    z = self.sample_z(template=z_tilde, sigma=self.z_sigma)

                    recon_loss = F.mse_loss(x_recon, x, size_average=False).div(self.batch_size)
                    mmd_loss = mmd(z_tilde, z, z_var=self.z_var)
                    total_loss = recon_loss + self._lambda*mmd_loss

                    self.optim.zero_grad()
                    total_loss.backward()
                    self.optim.step()

                    if self.global_iter%1000 == 0:
                        self.gather.insert(iter=self.global_iter,
                                        mu=z.mean(0).data, var=z.var(0).data,
                                        recon_loss=recon_loss.data, mmd_loss=mmd_loss.data,)

                    if self.global_iter%10000 == 0:
                        self.gather.insert(images=x.data)
                        self.gather.insert(images=x_recon.data)
                        self.viz_reconstruction()
                        self.viz_lines()
                        self.sample_x_from_z(n_sample=100)
                        self.gather.flush()
                        self.save_checkpoint('last')
                        pbar.write('[{}] total_loss:{:.3f} recon_loss:{:.3f} mmd_loss:{:.3f}'.format(
                            self.global_iter, total_loss.data.item(), recon_loss.data.item(), mmd_loss.data.item()))

                    if self.global_iter%60000 == 0:
                        self.save_checkpoint(str(self.global_iter))

                    if self.global_iter >= max_iter:
                        out = True
                        break

            pbar.write("[Training Finished]")

    def viz_reconstruction(self):
        self.net.eval()
        x = self.gather.data['images'][0][:100]
        x = make_grid(x, normalize=True, nrow=10)
        x_recon = F.sigmoid(self.gather.data['images'][1][:100])
        x_recon = make_grid(x_recon, normalize=True, nrow=10)
        images = torch.stack([x, x_recon], dim=0).cpu()
        self.viz.images(images, env=self.viz_name+'_reconstruction',
                        opts=dict(title=str(self.global_iter)), nrow=2)
        self.net.train()

    def viz_lines(self):
        self.net.eval()
        recon_losses = torch.stack(self.gather.data['recon_loss']).cpu()
        mmd_losses = torch.stack(self.gather.data['mmd_loss']).cpu()
        mus = torch.stack(self.gather.data['mu']).cpu()
        vars = torch.stack(self.gather.data['var']).cpu()
        iters = torch.Tensor(self.gather.data['iter'])

        legend = []
        for z_j in range(self.z_dim):
            legend.append('z_{}'.format(z_j))

        if self.win_recon is None:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))
        else:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.viz_name+'_lines',
                                        win=self.win_recon,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))

        if self.win_mmd is None:
            self.win_mmd = self.viz.line(
                                        X=iters,
                                        Y=mmd_losses,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='maximum mean discrepancy',))
        else:
            self.win_mmd = self.viz.line(
                                        X=iters,
                                        Y=mmd_losses,
                                        env=self.viz_name+'_lines',
                                        win=self.win_mmd,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='maximum mean discrepancy',))

        if self.win_mu is None:
            self.win_mu = self.viz.line(
                                        X=iters,
                                        Y=mus,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend,
                                            xlabel='iteration',
                                            title='posterior mean',))
        else:
            self.win_mu = self.viz.line(
                                        X=iters,
                                        Y=vars,
                                        env=self.viz_name+'_lines',
                                        win=self.win_mu,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend,
                                            xlabel='iteration',
                                            title='posterior mean',))

        if self.win_var is None:
            self.win_var = self.viz.line(
                                        X=iters,
                                        Y=vars,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend,
                                            xlabel='iteration',
                                            title='posterior variance',))
        else:
            self.win_var = self.viz.line(
                                        X=iters,
                                        Y=vars,
                                        env=self.viz_name+'_lines',
                                        win=self.win_var,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend,
                                            xlabel='iteration',
                                            title='posterior variance',))
        self.net.train()

    def sample_z(self, n_sample=None, dim=None, sigma=None, template=None):
        if n_sample is None:
            n_sample = self.batch_size
        if dim is None:
            dim = self.z_dim
        if sigma is None:
            sigma = self.z_sigma

        if template is not None:
            z = sigma*Variable(template.data.new(template.size()).normal_())
        else:
            z = sigma*torch.randn(n_sample, dim)
            z = Variable(z.cuda())

        return z

    def sample_x_from_z(self, n_sample):
        self.net.eval()
        z = self.sample_z(n_sample=n_sample, sigma=self.z_sigma)
        x_gen = F.sigmoid(self.net._decode(z)[:100]).data.cpu()
        x_gen = make_grid(x_gen, normalize=True, nrow=10)
        self.viz.images(x_gen, env=self.viz_name+'_sampling_from_random_z',
                        opts=dict(title=str(self.global_iter)))
        self.net.train()

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        win_states = {'recon':self.win_recon,
                      'mmd':self.win_mmd,
                      'mu':self.win_mu,
                      'var':self.win_var,}
        states = {'iter':self.global_iter,
                  'epoch':self.global_epoch,
                  'win_states':win_states,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = self.ckpt_dir.joinpath(filename)
        torch.save(states, file_path.open('wb+'))

        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename, silent=False):
        file_path = self.ckpt_dir.joinpath(filename)
        if file_path.is_file():
            checkpoint = torch.load(file_path.open('rb'), map_location="cpu")
            self.global_iter = checkpoint['iter']
            self.global_epoch = checkpoint['epoch']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_mmd = checkpoint['win_states']['mmd']
            self.win_var = checkpoint['win_states']['var']
            self.win_mu = checkpoint['win_states']['mu']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            if not silent:
                print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            if not silent:
                print("=> no checkpoint found at '{}'".format(file_path))
