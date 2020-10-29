import cv2
import numpy as np


def draw_img(img, bbox, gt_bbox=None, idx=0):
    pred_bbox = gen_pos(bbox)
    if gt_bbox is not None:
        gt_bbox = gen_pos(gt_bbox)

    red = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    img = cv2.rectangle(img, pred_bbox[0], pred_bbox[1], red, 2)
    if gt_bbox is not None:
        img = cv2.rectangle(img, gt_bbox[0], gt_bbox[1], green, 2)
    img = cv2.putText(img, str(idx), (5, 30), cv2.FONT_HERSHEY_DUPLEX, 1, yellow)

    return img


def gen_pos(bbox):
    min_x, min_y, w, h = bbox
    x0 = np.round(min_x).astype(int)
    y0 = np.round(min_y + h).astype(int)
    x1 = np.round(min_x + w).astype(int)
    y1 = np.round(min_y).astype(int)
    pos0, pos1 = (x0, y0), (x1, y1)

    return pos0, pos1
