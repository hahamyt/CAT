import numpy as np
import torch
from PIL import Image
import cv2
from skimage.exposure import exposure

from core.samples.aug.he import he

from core.samples.imaug import ImgAugmenter


class RegionExtractor():
    def __init__(self, image, samples, crop_size, padding, batch_size, shuffle=False, transform=None):

        self.image = np.asarray(image)
        self.samples = samples
        self.crop_size = crop_size
        self.padding = padding
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform

        self.index = np.arange(len(samples))
        self.pointer = 0

        self.mean = self.image.mean(0).mean(0).astype('float32')

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.samples):
            self.pointer = 0
            raise StopIteration
        else:
            next_pointer = min(self.pointer + self.batch_size, len(self.samples))
            index = self.index[self.pointer:next_pointer]
            self.pointer = next_pointer

            regions = self.extract_regions(index)
            # regions = torch.from_numpy(regions)
            return regions

    next = __next__

    def extract_regions(self, index):
        regions = torch.zeros((len(index), 3, self.crop_size, self.crop_size), dtype=torch.float32)

        for i, sample in enumerate(self.samples[index]):
            regions[i] = self.transform(crop_image(self.image, sample, self.crop_size, self.padding))

        return regions


class DataProvider():
    def __init__(self, image, samples, crop_size, padding, batch_size, shuffle=False, transform=None):
        self.image = np.asarray(image)
        self.samples = samples
        self.crop_size = crop_size
        self.padding = padding
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        if self.shuffle:
            np.random.shuffle(self.samples)

        self.index = np.arange(len(samples))
        self.pointer = 0

        self.mean = self.image.mean(0).mean(0).astype('float32')

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.samples):
            self.pointer = 0
            raise StopIteration
        else:
            next_pointer = min(self.pointer + self.batch_size, len(self.samples))
            index = self.index[self.pointer:next_pointer]
            self.pointer = next_pointer

            regions, labels = self.extract_regions(index)
            labels = torch.from_numpy(labels)
            regions = torch.from_numpy(regions)
            return regions, labels  # .unsqueeze(1).unsqueeze(2).unsqueeze(3)

    next = __next__

    def extract_regions(self, index):
        labels = np.zeros((len(index), 5)).astype(np.float32)
        regions = np.zeros((len(index), self.crop_size, self.crop_size, 3), dtype='uint8')
        # import matplotlib.pyplot as plt
        for i, sample in enumerate(self.samples[index]):
            regions[i] = crop_image(self.image, sample[:4], self.crop_size, self.padding)
            if len(sample) == 5:
                labels[i] = sample
            else:
                labels[i, :4] = sample
            # plt.imshow(regions[i])
            # plt.imsave("%d-%d.jpeg" % (labels[i], i), regions[i])
        pos_idx = np.where(labels[:, 4] == 1)
        pos_aug = np.array([ImgAugmenter(img).augment() for i in range(10) for img in regions[pos_idx]])  #
        # plt.imshow(pos_aug[1, :, :, :])
        aug_labels = np.tile(labels[pos_idx], (10, 1))
        if pos_aug.size != 0:
            regions = np.concatenate((regions, pos_aug), axis=0)
            labels = np.concatenate((labels, aug_labels), axis=0)
            # in order to have the same order for two numpy arrays
            state = np.random.get_state()
            np.random.shuffle(regions)
            np.random.set_state(state)
            np.random.shuffle(labels)

        regions = regions.transpose(0, 3, 1, 2).astype('float32')
        # regions = regions - 128.
        labels = np.array(labels).astype('float32')

        if self.transform is not None:
            for i in range(len(labels)):
                regions[i, :, :, :] = self.transform(
                    Image.fromarray(regions[i, :, :, :].transpose(1, 2, 0).astype('uint8')).convert('RGB'))

        return regions, labels.astype(np.float32)


def crop_image(img, bbox, img_size=64, padding=0, valid=False):
    # if np.random.rand() < 0.2:
    # img = Canny(img)
    #     # img = he(img)

    x, y, w, h = np.array(bbox, dtype='float32')

    half_w, half_h = w / 2, h / 2
    center_x, center_y = x + half_w, y + half_h

    if padding > 0:
        pad_w = padding * w / img_size
        pad_h = padding * h / img_size
        half_w += pad_w
        half_h += pad_h

    img_h, img_w, _ = img.shape
    min_x = int(center_x - half_w + 0.5)
    min_y = int(center_y - half_h + 0.5)
    max_x = int(center_x + half_w + 0.5)
    max_y = int(center_y + half_h + 0.5)

    if valid:
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(img_w, max_x)
        max_y = min(img_h, max_y)

    if min_x >= 0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)

        cropped = 128 * np.ones((max_y - min_y, max_x - min_x, 3), dtype='uint8')
        cropped[min_y_val - min_y:max_y_val - min_y, min_x_val - min_x:max_x_val - min_x, :] \
            = img[min_y_val:max_y_val, min_x_val:max_x_val, :]
    # print(cropped.shape)
    scaled = cv2.resize(cropped, (img_size, img_size))
    return Image.fromarray(scaled.astype('uint8')).convert('RGB')


def Canny(img, lowThreshold=20):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ratio = 3
    kernel_size = 3
    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
    detected_edges = cv2.Canny(detected_edges,
                               lowThreshold,
                               lowThreshold * ratio,
                               apertureSize=kernel_size)
    dst = cv2.bitwise_and(img, img, mask=detected_edges)  # just add some colours to edges from original image.
    # cv2.imshow('canny demo', dst)
    # cv2.waitKey(1)
    return dst
