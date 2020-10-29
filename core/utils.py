import cv2
import torch
import numpy as np
import torchvision

from net.wae64 import WAE64
from tracking.options import opts


def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''

    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]

    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


def bbreg_boxes(reg, bbox, img_sz):
    reg = reg.cpu().detach().numpy()
    bbox = bbox.cpu().detach().numpy()
    bbox_ = bbox

    bbox_[:, :2] = bbox_[:, :2] + bbox_[:, 2:] / 2
    bbox_[:, :2] = reg[:, :2] * bbox_[:, 2:] + bbox_[:, :2]
    bbox_[:, 2:] = np.exp(reg[:, 2:]) * bbox_[:, 2:]
    bbox_[:, :2] = bbox_[:, :2] - bbox_[:, 2:] / 2

    r = overlap_ratio(bbox, bbox_)
    s = np.prod(bbox[:, 2:], axis=1) / np.prod(bbox_[:, 2:], axis=1)
    idx = (r >= opts['overlap_bbreg'][0]) * (r <= opts['overlap_bbreg'][1]) * \
          (s >= opts['scale_bbreg'][0]) * (s <= opts['scale_bbreg'][1])
    idx = np.logical_not(idx)
    bbox_[idx] = bbox[idx]

    bbox_[:, :2] = np.maximum(bbox_[:, :2], 0)
    bbox_[:, 2:] = np.minimum(bbox_[:, 2:], img_sz - bbox[:, :2])

    return torch.from_numpy(bbox_)


def tranxyxy2xywh(boxes):
    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    bx_min = boxes[:, 0]
    by_min = boxes[:, 1]
    return np.vstack((bx_min, by_min, bw, bh)).transpose(1, 0)


def shuffleTensor(pos_feats, neg_feats, params):
    feats = torch.cat((pos_feats, neg_feats)).cpu().detach().numpy()
    targets = torch.cat((torch.ones((len(pos_feats), 1)),
                         torch.zeros((len(neg_feats), 1)))).cpu().detach().numpy()

    pos_ious = params[0]  # 下周3 14号 9:30 南石马路一号
    neg_ious = params[1]
    mask = np.concatenate((pos_ious, neg_ious))

    state = np.random.get_state()
    np.random.shuffle(feats)
    np.random.set_state(state)
    np.random.shuffle(targets)
    np.random.set_state(state)
    np.random.shuffle(mask)

    feats = torch.from_numpy(feats).cuda()
    mask = torch.from_numpy(mask).float().cuda()
    # One hot labels
    targets = torch.LongTensor(targets)
    # targets = torch.zeros(len(targets), 2, dtype=torch.long).scatter_(1, targets, 1).cuda()
    targets = torch.zeros(len(targets), 2).scatter_(1, targets, 1).cuda()
    # trans label [0, 1] to [-1, 1]
    # targets[targets == 0] = -1
    targets[:, 1] = targets[:, 1] * mask
    targets[:, 0] = targets[:, 0] * (1 - mask)
    # targets[:, 0] = 1 - targets[:, 1]
    return feats, targets


def forward_feats(model, regions, batch_sz=opts['batch_extractor'], sift=False):
    model.eval()
    num_iter = len(regions) // batch_sz
    ptr = 0
    all_feats = []
    if num_iter == 0:
        if sift:
            all_feats = model(((regions[:, 0, :, :] * 19595 + regions[:, 1, :, :] * 38469
                                + regions[:, 2, :, :] * 7472) >> 16).unsqueeze(1)).detach()
        else:
            all_feats = model(regions).detach()
    else:
        for i in range(num_iter):
            if sift:
                all_feats.append(
                    model(((regions[ptr:(ptr + batch_sz), 0, :, :] * 19595 + regions[ptr:(ptr + batch_sz), 1, :,
                                                                             :] * 38469
                            + regions[ptr:(ptr + batch_sz), 2, :, :] * 7472) >> 16).unsqueeze(1)).detach())
            else:
                all_feats.append(model(regions[ptr:(ptr + batch_sz)]).detach())
            ptr += batch_sz
            if 0 < len(regions) - ptr < batch_sz:
                if sift:
                    all_feats.append(
                        model(((regions[ptr:, 0, :, :] * 19595 + regions[ptr:, 1, :, :] * 38469
                                + regions[ptr:, 2, :, :] * 7472) >> 16).unsqueeze(1)).detach())
                else:
                    all_feats.append(model(regions[ptr:]).detach())
                break
        all_feats = torch.cat(all_feats)
    return all_feats


def load_weights(ckpt_path, model):
    checkpoint = torch.load(ckpt_path)
    pretrain_dict = checkpoint['model_states']['net']

    weights = {}
    for name, params in pretrain_dict.items():
        encoder_name = name.split('.')[0]
        conv_name = name.split('.')[1]
        rest_name = name.split(encoder_name)[-1]
        layer_name = encoder_name + rest_name

        if encoder_name == 'encoder':
            if conv_name in ['conv1', 'conv2', 'conv3']:
                encoder_name = 'encoder1'
            else:
                encoder_name = 'encoder2'
            layer_name = encoder_name + rest_name
        elif encoder_name == 'score':
            continue
        weights[layer_name] = params

    model.load_state_dict(weights)

# sample many rects around a center point center with radicus r
def sample_bbox(center, sample_sz, r, n_samples):
    cx, cy = center
    w, h = sample_sz

    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    sx, sy = cx + r * np.cos(theta), cy + r * np.sin(theta)
    x_min, y_min = np.int32(sx - w / 2 + 0.5), np.int32(sy - h / 2 + 0.5)
    x_max, y_max = x_min + w, y_min + h
    rects = np.stack((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)
    return rects
