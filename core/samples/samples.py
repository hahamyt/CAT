#!/usr/bin/python3
#
# This file is created to establish the feature's fetching an clustering
#
import os
import sys

import Augmentor
import cv2
from PIL import Image

sys.path.append(os.path.abspath('../'))
import numpy as np
import torch
from torchvision.transforms import transforms
from core.samples.sample_generator import gen_bboxes, SampleGenerator
from tracking.options import opts
from core.samples.region2imgs import crop_image
from core.samples.gmm_manager import postive_samples_manager, negative_samples_manger
from core.utils import forward_feats, overlap_ratio
import multiprocessing

##
#   This class is created to manage samples
##

class Memory():
    def __init__(self, extractor):
        self.neg_features = []                            # where negative features stored
        self.extractor = extractor                 # feature extractor
        # data argument
        p = Augmentor.Pipeline()
        # if opts["seed"] is not None:
        #     p.set_seed(opts["seed"])
        # p.rotate(probability=0.1, max_right_rotation=10, max_left_rotation=0)
        # p.random_distortion(probability=0.1, grid_width=2, grid_height=2, magnitude=4)
        # p.flip_left_right(probability=0.2)
        # p.flip_top_bottom(probability=0.1)
        # p.random_erasing(probability=0.01, rectangle_area=0.2)
        # p.flip_random(probability=0.1)
        p.resize(probability=1.0, width=opts['img_size'], height=opts['img_size'])
        self.transform = transforms.Compose([
            p.torch_transform(),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.pos_manager = postive_samples_manager(pos_num=opts["samplespace_pos"] )
        self.neg_manger = negative_samples_manger(neg_num=opts["samplespace_neg"] )
        self.frames = 0
        self.neg_ious = []

    def init(self, frame, bbox):
        self.size = (frame.shape[1], frame.shape[0])
        # for update
        self.pos_generator = SampleGenerator('gaussian', self.size, opts['trans_pos'], opts['scale_pos'])
        self.neg_generator = SampleGenerator('uniform', self.size, opts['trans_neg'], opts['scale_neg'])

        # generate bounding box rect
        pos_rects = gen_bboxes(SampleGenerator('gaussian', self.size, opts['trans_pos'], opts['scale_pos']),
                               bbox, opts['n_pos_init'], opts['overlap_pos_init'])

        overlap_pos = opts['overlap_pos_init'].copy()
        while len(pos_rects) == 0 and overlap_pos[0] > 0.3:
            overlap_pos[0] -= 0.1
            pos_rects = gen_bboxes(SampleGenerator('gaussian', self.size, opts['trans_pos'], opts['scale_pos']),
                                   bbox, opts['n_pos_init'], overlap_pos)

        neg_rects = np.random.permutation(np.concatenate([
            gen_bboxes(SampleGenerator('uniform', self.size, opts['trans_neg_init'], opts['scale_neg_init']),
                       bbox, opts['n_neg_init'] // 2, opts['overlap_neg_init']),
            gen_bboxes(SampleGenerator('whole', self.size),
                       bbox, opts['n_neg_init'] // 2, opts['overlap_neg_init'])]))
        neg_rects = np.random.permutation(neg_rects)

        # Extract pos/neg features
        pos_feats = self.extract_feats(frame, pos_rects)
        neg_feats = self.extract_feats(frame, neg_rects)

        # generate corrsponding features of these rects
        self.pos_manager.insert(pos_feats, overlap_ratio(pos_rects, bbox))

        self.neg_manger.insert(neg_feats, overlap_ratio(neg_rects, bbox))

        self.frames += 1

    def del_features(self, n_pos, n_neg):
        self.pos_manager.pos_features[0] = self.pos_manager.pos_features[0][:n_pos]
        self.pos_manager.pos_ious[0] = self.pos_manager.pos_ious[0][:n_pos]
        self.neg_manger.neg_features[0] = self.neg_manger.neg_features[0][:n_neg]
        self.neg_manger.neg_ious[0] = self.neg_manger.neg_ious[0][:n_neg]

    def add_features(self, frame, rect):
        # frame = np.asarray(frame)
        pos_rects = gen_bboxes(self.pos_generator, rect, opts['n_pos_update'], opts['overlap_pos_update'])  # 200

        overlap_pos = opts['overlap_pos_update'].copy()
        while len(pos_rects) == 0 and overlap_pos[0] > 0.3:
            overlap_pos[0] -= 0.1
            pos_rects = gen_bboxes(self.pos_generator, rect, opts['n_pos_update'], overlap_pos)

        neg_rects = np.random.permutation(np.concatenate([
            gen_bboxes(self.neg_generator, rect, opts['n_neg_update'] - opts['n_neg_update'] // 6, opts['overlap_neg_update']),
            gen_bboxes(SampleGenerator('whole', self.size),rect, opts['n_neg_update'] // 6, [0, 0])
        ]))

        # Extract pos/neg features
        pos_feats = self.extract_feats(frame, pos_rects)
        neg_feats = self.extract_feats(frame, neg_rects)

        self.pos_manager.insert(pos_feats, overlap_ratio(pos_rects, rect))
        self.neg_manger.insert(neg_feats, overlap_ratio(neg_rects, rect))

        self.frames += 1

    def get_samples(self):
        return self.pos_manager.pos_features, self.neg_manger.neg_features, self.pos_manager.pos_ious, self.neg_manger.neg_ious

    # get all init and new added features
    def get_features(self, type='lupdate', samples=2):
        if type == 'lupdate':
            return torch.cat(self.pos_manager.pos_features, 0), torch.cat(self.neg_manger.neg_features, 0), self.pos_manager.pos_ious, self.neg_manger.neg_ious
        elif type == 'supdate':
            total_neg = len(self.neg_manger.neg_features)

            if samples > total_neg:
                samples = total_neg

            pos_feats = self.pos_manager.pos_features.copy()
            pos_ious = self.pos_manager.pos_ious.copy()
            pos_feats = torch.cat(pos_feats[-samples:], 0)
            pos_ious = pos_ious[-samples:]

            neg_feats = torch.cat(self.neg_manger.neg_features[-samples:], 0)
            neg_ious = self.neg_manger.neg_ious[-samples:]
            return pos_feats, neg_feats, pos_ious, neg_ious

    def get_bbreg_feats(self, img_sz, init_bbox, first_frame):
        # Train bbox regressor
        bbreg_rects = gen_bboxes(SampleGenerator('uniform', img_sz, 0.3, 1.5, 1.1),
                                 init_bbox, opts['n_bbreg'], opts['overlap_bbreg'], opts['scale_bbreg'])

        overlap_bbreg = opts['overlap_bbreg'].copy()
        while len(bbreg_rects) == 0 and overlap_bbreg[0] > 0.3:
            overlap_bbreg[0] -= 0.1
            print('resample bbreg rects, min_iou {}'.format(overlap_bbreg[0]))
            bbreg_rects = gen_bboxes(SampleGenerator('uniform', img_sz, 0.3, 1.5, 1.1),
                                     init_bbox, opts['n_bbreg'], overlap_bbreg, opts['scale_bbreg'])

        bbreg_feats = self.extract_feats(first_frame, bbreg_rects)
        return bbreg_rects, bbreg_feats

    # extract rects' features
    def extract_feats(self, frame, rects):
        with torch.no_grad():
            # frame = np.asanyarray(frame)
            regions = self.extract_regions(frame, rects).cuda()
            batch_sz = opts['batch_extractor']

            feats = forward_feats(self.extractor, regions, batch_sz)

            return feats

    # extract frames' regions of every rect
    def extract_regions(self, frame, rects, toTensor=True):
        regions = []
        for r in rects:
            if toTensor:
                regions.append(self.transform(crop_image(frame, r, img_size=opts['img_size'], padding=opts['padding'])).unsqueeze(0))
            else:
                regions.append(crop_image(frame, r, img_size=opts['img_size'], padding=opts['padding']).unsqueeze(0))
        regions = torch.cat(regions)
        return regions
