#!/usr/bin/python3
import os
import sys
from core.trackers.tracker_fchf import DCFT_FCHF as DCFT
sys.path.append(os.path.abspath('../'))
import time
import cv2
from core.utils import overlap_ratio
import argparse
import torch
from core.drawer import draw_img
from tracking.options import opts
import numpy as np
from tracking.gen_config import gen_config

os.environ['CUDA_VISIBLE_DEVICES'] = opts['gpu']

if __name__ == '__main__':
    if opts['seed'] is not None:
        np.random.seed(opts['seed'])
        torch.manual_seed(opts['seed'])
        torch.cuda.manual_seed(opts['seed'])
        torch.cuda.manual_seed_all(opts['seed'])
        torch.backends.cudnn.deterministic = True

    tic = time.time()
    parser = argparse.ArgumentParser()
    # Singer2 and MotorRolling and Skiing and Trans(last frames)
    parser.add_argument('-s', '--seq', default=opts['seq_name'], help='input seq')
    parser.add_argument('-j', '--json', default='', help='input json')
    parser.add_argument('-f', '--savefig', default=False, action='store_true')
    parser.add_argument('-d', '--display', default=False, action='store_true')
    parser.add_argument('-t', '--type', default="OTB")
    args = parser.parse_args()
    assert (args.seq != '' or args.json != '')

    # Generate sequence config
    img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)

    first_frame = cv2.cvtColor(cv2.imread(img_list[0]), cv2.COLOR_BGR2RGB)

    tracker = DCFT(frame=first_frame, init_bbox=init_bbox, gt=gt)

    checkpoint = torch.load(opts["ckpt_path"])
    pretrain_dict = checkpoint['model_states']['net']
    model_dict = tracker.encoder.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict
                     and model_dict[k].size() == v.size()}
    tracker.encoder.load_state_dict(pretrain_dict)
    tracker.encoder.__delattr__('decoder')  # For saving memory
    torch.cuda.empty_cache()

    tracker.init()

    all_iou = []
    result_bb = []

    spf_total = time.time() - tic
    for i in range(1, len(img_list)):
        tic = time.time()
        try:
            frame = cv2.cvtColor(cv2.imread(img_list[i]), cv2.COLOR_BGR2RGB)
        except:
            break
        rect, bbreg_rect, score = tracker.update(frame=frame)
        result_bb.append(bbreg_rect)
        torch.cuda.empty_cache()
        iou = overlap_ratio(bbreg_rect, gt[i])[0]
        all_iou.append(iou)

        print("frame: {}/{}, score: {:.3f} iou: {:.3f}".format(i, len(img_list) - 1, score, iou.squeeze()))

        if opts['vis']:
            img = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            img = draw_img(img, bbreg_rect, gt[i], idx=i)
            cv2.imshow('tracker', img)
            cv2.waitKey(1)
            cv2.imwrite('../result/{}/{}.'.format('Bird1', str(i).zfill(4)), draw_img(img, bbreg_rect, gt[i], idx=i))

        spf = time.time() - tic
        spf_total += spf

    fps = i / spf_total
    print("mean iou: {:.3f} fps: {:.3f}".format(np.mean(all_iou), fps))