"""
The implementation of GHM-C and GHM-R losses.
Details can be found in the paper `Gradient Harmonized Single-stage Detector`:
https://arxiv.org/abs/1811.05181

Copyright (c) 2018 Multimedia Laboratory, CUHK.
Licensed under the MIT License (see LICENSE for details)
Written by Buyu Li
"""

import torch
import torch.nn.functional as F

from tracking.options import opts
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


class GHMC_Loss:
    def __init__(self, bins=10, momentum=0):
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins+1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]

    def calc(self, input, target, ):                # mask):
        """ Args:
        input [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary target (0 or 1) for each sample each class. The value is -1
            when the sample is ignored.
        """
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(input)

        # gradient length
        g = torch.abs(input.sigmoid().detach() - target)

        # valid = mask > 0
        tot = 1.0                                               # max(valid.float().sum().item(), 1.0)
        n = 0                                                   # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1])                # & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            input, target, weights, reduction='sum') / tot
        return loss


class GHMR_Loss:
    def __init__(self, mu=0.02, bins=10, momentum=0):
        self.mu = mu
        self.bins = bins
        self.edges = [float(x) / bins for x in range(bins+1)]
        self.edges[-1] = 1e3
        self.momentum = momentum
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]

    def calc(self, input, target,):                                  #  mask):
        """ Args:
        input [batch_num, 4 (* class_num)]:
            The prediction of box regression layer. Channel number can be 4 or
            (4 * class_num) depending on whether it is class-agnostic.
        target [batch_num, 4 (* class_num)]:
            The target regression values with the same size of input.
        """
        mu = self.mu
        edges = self.edges
        mmt = self.momentum

        # ASL1 loss
        diff = input - target
        loss = torch.sqrt(diff * diff + mu * mu) - mu

        # gradient length
        g = torch.abs(diff / torch.sqrt(mu * mu + diff * diff)).detach()
        weights = torch.zeros_like(g)
        # # 设置matplotlib正常显示中文和负号
        # matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
        # matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        # plt.rcParams['savefig.dpi'] = 150  # 图片像素
        # plt.rcParams['figure.dpi'] = 150  # 分辨率
        #
        # data = g.cpu().numpy()[:, 1]
        # """
        # 绘制直方图
        # data:必选参数，绘图数据
        # bins:直方图的长条形数目，可选项，默认为10
        # normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
        # facecolor:长条形的颜色
        # edgecolor:长条形边框的颜色
        # alpha:透明度
        # """
        # plt.hist(data, bins=100, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
        # # 显示横轴标签
        # plt.xlabel("gradient norm", fontsize=16)
        # # 显示纵轴标签
        # plt.ylabel("value", fontsize=16)
        # # plt.xlim(0, 0.04)
        # plt.ylim(0, 50)
        # # 显示图标题
        # # plt.title("频数/频率分布直方图")
        # np.savetxt("gradient_norm.csv", data, delimiter=',')
        # plt.show()
        # valid = mask > 0
        tot = 1.0                               # max(mask.float().sum().item(), 1.0)
        n = 0  # n: valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1])                   # & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                n += 1
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
        if n > 0:
            weights /= n

        loss = loss * weights
        loss = loss.sum() / tot
        return loss


class GHM_MSE_Loss:
    def __init__(self, mu=0.02, bins=10, momentum=0):
        self.mu = mu
        self.bins = bins
        self.edges = [float(x) / bins for x in range(bins+1)]
        self.edges[-1] = 1e3
        self.momentum = momentum
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]

    def calc(self, input, target,):                                  #  mask):
        """ Args:
        input [batch_num, 4 (* class_num)]:
            The prediction of box regression layer. Channel number can be 4 or
            (4 * class_num) depending on whether it is class-agnostic.
        target [batch_num, 4 (* class_num)]:
            The target regression values with the same size of input.
        """
        mu = self.mu
        edges = self.edges
        mmt = self.momentum

        # MSE loss
        diff = input - target
        loss = diff.pow(2)/(2 * diff.shape[0] * diff.shape[1])

        # gradient length
        g = torch.abs(diff / diff.shape[0]).detach()
        weights = torch.zeros_like(g)
        # if opts["debug"]:

        # 设置matplotlib正常显示中文和负号
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
        matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        plt.rcParams['savefig.dpi'] = 150  # 图片像素
        plt.rcParams['figure.dpi'] = 150  # 分辨率

        data = g.cpu().numpy()[:, 1]
        """
        绘制直方图
        data:必选参数，绘图数据
        bins:直方图的长条形数目，可选项，默认为10
        normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
        facecolor:长条形的颜色
        edgecolor:长条形边框的颜色
        alpha:透明度
        """
        plt.hist(data, bins=10, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
        # 显示横轴标签
        plt.xlabel("gradient norm", fontsize=16)
        # 显示纵轴标签
        plt.ylabel("value", fontsize=16)
        # plt.xlim(0, 0.04)
        plt.ylim(0, 50)
        # 显示图标题
        # plt.title("频数/频率分布直方图")
        np.savetxt("gradient_norm.csv", data, delimiter=',')
        plt.show()

        # valid = mask > 0
        tot = 1.0                               # max(mask.float().sum().item(), 1.0)
        n = 0  # n: valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1])                   # & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                n += 1
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
        if n > 0:
            weights /= n

        loss = loss * weights
        loss = loss.sum() / tot
        return loss