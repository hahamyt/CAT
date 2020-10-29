from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.init as init

from tracking.options import opts


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class WAE64(nn.Module):
    def __init__(self, tr=False):
        super(WAE64, self).__init__()
        self.tr = tr
        self.encoder = nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(
                    nn.Conv2d(3, 128, 4, 2, 1, bias=False),  # B,  128, 32, 32
                    nn.BatchNorm2d(128),
                    nn.ReLU(True))),
            ('conv2', nn.Sequential(
                    nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # B,  256, 16, 16
                    nn.BatchNorm2d(256),
                    nn.ReLU(True))),
            ('conv3', nn.Sequential(
                    nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # B,  512,  8,  8
                    nn.BatchNorm2d(512),
                    nn.ReLU(True))),
            ('conv4', nn.Sequential(
                    nn.Conv2d(512, 1024, 4, 2, 1, bias=False),  # B, 1024,  4,  4
                    nn.BatchNorm2d(1024),
                    nn.ReLU(True),
                    View((-1, 1024 * 4 * 4)),  # B, 1024*4*4
                    nn.Linear(1024 * 4 * 4, opts['in_dim']),
                    # nn.Tanh(),
                    ))  # B, z_dim
        ]))

        # self.encoder1 = nn.Sequential(OrderedDict([
        #         ('conv1', nn.Sequential(
        #                 nn.Conv2d(3, 128, 4, 2, 1, bias=False),  # B,  128, 32, 32
        #                 nn.BatchNorm2d(128),
        #                 nn.ReLU(True))),
        #         ('conv2', nn.Sequential(
        #             nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # B,  256, 16, 16
        #             nn.BatchNorm2d(256),
        #             nn.ReLU(True))),
        #         ('conv3', nn.Sequential(
        #             nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # B,  512,  8,  8
        #             nn.BatchNorm2d(512),
        #             nn.ReLU(True))),
        # ]))
        # self.encoder2 = nn.Sequential(OrderedDict([
        #         ('conv4', nn.Sequential(
        #                 nn.Conv2d(512, 1024, 4, 2, 1, bias=False),  # B, 1024,  4,  4
        #                 nn.BatchNorm2d(1024),
        #                 nn.ReLU(True),
        #                 View((-1, 1024 * 4 * 4)),  # B, 1024*4*4
        #                 nn.Linear(1024 * 4 * 4, 128)))  # B, z_dim
        # ]))
        #
        # self.maxpool = nn.MaxPool2d(8)

        self.decoder = nn.Sequential(OrderedDict([
            ('deconv1',nn.Sequential(
                    nn.Linear(opts['in_dim'], 1024 * 8 * 8),  # B, 1024*8*8
                    View((-1, 1024, 8, 8)),  # B, 1024,  8,  8
                    nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),  # B,  512, 16, 16
                    nn.BatchNorm2d(512),
                    nn.ReLU(True))),
            ('deconv2',nn.Sequential(
                    nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # B,  256, 32, 32
                    nn.BatchNorm2d(256),
                    nn.ReLU(True))),
            ('deconv3',nn.Sequential(
                    nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # B,  128, 64, 64
                    nn.BatchNorm2d(128),
                    nn.ReLU(True))),
            ('deconv4',nn.Sequential(
                nn.ConvTranspose2d(128, 3, 1))),  # B,   nc, 64, 64
        ]))

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            if block == 'maxpool':
                continue
            for m in self._modules[block]:
                for i in m:
                    kaiming_init(i)

    def forward(self, x):
        if self.tr:
            noise = torch.normal(torch.zeros_like(x.data), std=0.01)
            x.data += noise
            z = self._encode(x)
            x_recon = self._decode(z)
            return x_recon, z
        else:
            return self._encode(x)
            # x1 = self.encoder1(x)
            # x1_ = self.maxpool(x1).squeeze()
            # x2 = self.encoder2(x1)
            # x2 = torch.cat([x1_, x2], dim=1)
            # return x2

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

    def _score(self, z):
        return self.score(z)

class Adversary(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10):
        super(Adversary, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),                                # B, 512
            nn.ReLU(True),
            nn.Linear(512, 512),                                  # B, 512
            nn.ReLU(True),
            nn.Linear(512, 512),                                  # B, 512
            nn.ReLU(True),
            nn.Linear(512, 512),                                  # B, 512
            nn.ReLU(True),
            nn.Linear(512, 1),                                    # B,   1
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

def append_params(params, module, prefix):
    for child in module.children():
        if isinstance(child, View):
            continue
        for k, p in child._parameters.items():
            if p is None: continue

            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            elif isinstance(child, nn.Linear):
                name = prefix + '_linear_' + k
            else:
                name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError("Duplicated param name: %s" % (name))