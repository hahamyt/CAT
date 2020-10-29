import os
import json
import numpy as np

def gen_config(args):

    if args.seq != '':
        # generate config from a sequence name
        if args.type == "OTB":
            seq_home = '../data/OTB'
            seq_name = args.seq
            img_dir = os.path.join(seq_home, seq_name, 'img')
        elif args.type == "VOT":
            seq_home = '../data/VOT'
            seq_name = args.seq
            img_dir = os.path.join(seq_home, seq_name)
        save_home = '../result_fig'
        result_home = '../result'

        img_list = os.listdir(img_dir)
        img_list.sort()
        img_list = [os.path.join(img_dir,x) for x in img_list]
        if args.type == "OTB":
            gt_path = os.path.join(seq_home, seq_name, 'groundtruth_rect.txt')
            try:
                gt = np.loadtxt(gt_path, delimiter=',')
            except ValueError:
                gt = np.loadtxt(gt_path)
            gt[:, :2] -= 1
            init_bbox = gt[0]
        elif args.type == "VOT":
            gt_path = os.path.join(seq_home, seq_name, 'groundtruth.txt')
            try:
                gt = np.loadtxt(gt_path, delimiter=',')
            except ValueError:
                gt = np.loadtxt(gt_path)
            center_x = (gt[:, 0] + gt[:, 2] + gt[:, 4] + gt[:, 6]) / 4
            center_y = (gt[:, 1] + gt[:, 3] + gt[:, 5] + gt[:, 7]) / 4
            min_x = np.amin(np.stack((gt[:, 0], gt[:, 2], gt[:, 4], gt[:, 6]), axis=1), axis=1)
            min_y = np.amin(np.stack((gt[:, 1], gt[:, 3], gt[:, 5], gt[:, 7]), axis=1), axis=1)
            w = 2 * (center_x - min_x)
            h = 2 * (center_y - min_y)

            gt = np.stack((min_x, min_y, w, h), axis=1)

            init_bbox = gt[0]

        savefig_dir = os.path.join(save_home,seq_name)
        result_dir = os.path.join(result_home,seq_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir,'result.json')

    elif args.json != '':
        # load config from a json file

        param = json.load(open(args.json,'r'))
        seq_name = param['seq_name']
        img_list = param['img_list']
        init_bbox = param['init_bbox']
        savefig_dir = param['savefig_dir']
        result_path = param['result_path']
        gt = None
        
    if args.savefig:
        if not os.path.exists(savefig_dir):
            os.makedirs(savefig_dir)
    else:
        savefig_dir = ''

    return img_list, init_bbox, gt, savefig_dir, args.display, result_path


def gen_config__(seq_name):

    # generate config from a sequence name

    seq_home = '../dataset/OTB'
    save_home = '../result_fig'
    result_home = '../result'

    img_dir = os.path.join(seq_home, seq_name, 'img')
    gt_path = os.path.join(seq_home, seq_name, 'groundtruth_rect.txt')

    img_list = os.listdir(img_dir)
    img_list.sort()
    img_list = [os.path.join(img_dir, x) for x in img_list]

    gt = np.loadtxt(gt_path, delimiter=',')
    init_bbox = gt[0]

    savefig_dir = os.path.join(save_home, seq_name)
    result_dir = os.path.join(result_home, seq_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_path = os.path.join(result_dir, 'result.json')
    return img_list, init_bbox, gt, savefig_dir, result_path

def gen_imglist(seq_path):
    img_list = os.listdir(seq_path)
    img_list.sort()
    img_list = [os.path.join(seq_path, x) for x in img_list]
    return img_list
