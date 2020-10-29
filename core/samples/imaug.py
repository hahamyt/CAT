import numpy as np
import cv2
import scipy, scipy.misc, scipy.signal
import torch


class ImgAugmenter(object):  # Only support 3 channel images for input now
    def __init__(self, img):
        self.img = img
        self.img_sz = img.shape[:2]

    def augment(self, illum_ratio=(0.5, 1.5), motion_blur_ratio=(0.05, 0.15),
                avg_blur_ratio=(0.05, 0.15), rotation_angle=(-60, 60),
                shift_ratio=(-0.2, 0.2), rescale_ratio=(0.5, 2),
                noise_ratio=(0.005, 0.01), occlusion_ratio=(0.25, 0.3)):

        augment_idx = np.random.choice(8)

        if augment_idx == 0: self.illumination(illum_ratio)
        elif augment_idx == 1: return self.motion_blur(motion_blur_ratio)
        elif augment_idx == 2: return self.rotation(rotation_angle) # self.avg_blur(avg_blur_ratio)
        elif augment_idx == 3: return self.rotation(rotation_angle)
        elif augment_idx == 4: return self.rotation(rotation_angle) # self.translation(shift_ratio)
        elif augment_idx == 5: return self.rotation(rotation_angle) # self.rescale(rescale_ratio)
        elif augment_idx == 6: return self.rotation(rotation_angle) # self.random_noise(noise_ratio)
        elif augment_idx == 7: return self.rotation(rotation_angle) # self.occlusion(occlusion_ratio)

    def illumination(self, gamma):  # the larger, the darker
        min_gamma, max_gamma = gamma
        gamma = np.random.uniform(min_gamma, max_gamma)
        table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)])

        return cv2.LUT(self.img, table)

    def motion_blur(self, kernel_sz_ratio):
        min_ratio, max_ratio = kernel_sz_ratio
        ratio = np.random.uniform(min_ratio, max_ratio)

        kernel_sz = np.ceil(min(self.img_sz) * ratio).astype(int)
        blur_kernel = np.zeros((kernel_sz, kernel_sz))
        blur_kernel[int((kernel_sz - 1) / 2), :] = np.ones(kernel_sz)
        blur_kernel = blur_kernel / kernel_sz

        return cv2.filter2D(self.img, -1, blur_kernel)

    def avg_blur(self, kernel_sz_ratio):
        min_ratio, max_ratio = kernel_sz_ratio
        ratio = np.random.uniform(min_ratio, max_ratio)
        kernel_sz = np.ceil(min(self.img_sz) * ratio).astype(int)

        return cv2.blur(self.img, (kernel_sz, kernel_sz))

    def rotation(self, angle):
        min_angle, max_angle = angle
        angle = np.random.uniform(min_angle, max_angle)

        center = tuple(np.array(self.img_sz) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1)

        return cv2.warpAffine(self.img, rot_mat, self.img_sz, flags=cv2.INTER_LINEAR)

    def translation(self, shift_ratio):
        min_ratio, max_ratio = shift_ratio
        shift_x = np.random.uniform(min_ratio, max_ratio)
        shift_y = np.random.uniform(min_ratio, max_ratio)

        h, w = self.img_sz
        x = shift_x * w
        y = shift_y * h
        trans_matrix = np.float32([[1, 0, x], [0, 1, y]])

        return cv2.warpAffine(self.img, trans_matrix, (w, h))

    def rescale(self, aspect_ratio):
        min_ratio, max_ratio = aspect_ratio
        ratio = np.random.uniform(min_ratio, max_ratio)

        h, w = self.img_sz
        w_scale = int(w * ratio)
        max_sz = max(h, w_scale)
        min_sz = min(h, w_scale)
        img_scale = cv2.resize(self.img, (w_scale, h))
        pad_sz = (max_sz - min_sz) // 2

        if w_scale > h:
            pad_img = np.zeros((pad_sz, max_sz, 3))
            img_new = np.concatenate((pad_img, img_scale, pad_img), axis=0)
            img_new = img_new.astype(np.uint8)
        else:
            pad_img = np.zeros((max_sz, pad_sz, 3))
            img_new = np.concatenate((pad_img, img_scale, pad_img), axis=1)
            img_new = img_new.astype(np.uint8)

        return cv2.resize(img_new, self.img_sz)

    def random_noise(self, noise_ratio, s_vs_p=0.5):
        min_ratio, max_ratio = noise_ratio
        ratio = np.random.uniform(min_ratio, max_ratio)
        img_new = self.img.copy()

        # Salt mode
        num_salt = np.ceil(ratio * self.img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in self.img.shape]
        img_new[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(ratio * self.img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in self.img.shape]
        img_new[coords] = 0

        return img_new

    def occlusion(self, sz_ratio):
        min_sz_ratio, max_sz_ratio = sz_ratio
        ratio = np.random.uniform(min_sz_ratio, max_sz_ratio)
        sz = int(ratio * min(self.img_sz))

        h, w = self.img_sz
        x_max, y_max = w - sz, h - sz
        x = int(np.random.uniform(0, x_max))
        y = int(np.random.uniform(0, y_max))

        img_new = self.img.copy()
        img_new[y:(y + sz), x:(x + sz)] = 0

        return img_new


def sample_aug_imgs(imgs, mean, std):
    imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1)
    imgs = np.uint8((imgs * std + mean) * 255)
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0

    aug_idxes = np.random.choice(np.arange(len(imgs)), len(imgs) // 2, replace=False)
    raw_idxes = np.delete(np.arange(len(imgs)), aug_idxes)
    aug_imgs = np.array([ImgAugmenter(img).augment() for img in imgs[aug_idxes]])
    raw_imgs = imgs[raw_idxes]

    batch_imgs = np.concatenate((raw_imgs, aug_imgs), axis=0)
    batch_imgs = np.random.permutation(batch_imgs)
    batch_imgs = ((batch_imgs / 255 - mean) / std).transpose(0, 3, 1, 2)

    return torch.Tensor(batch_imgs)
