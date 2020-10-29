import os

import cv2
import numpy as np
import torch
import torchvision

from net.wae64 import WAE64
from tracking.options import opts

BIGNUM = 1e22         # e22

class postive_samples_manager():
    def __init__(self, pos_num):
        # postive samples
        self.pos_features = []
        # Distance matrix stores the square of the euclidean distance between each pair of samples. Initialise it to inf
        self.distance_matrix = torch.ones((pos_num, pos_num), dtype=torch.float32).cuda() * BIGNUM
        # Kernel matrix, used to update distance matrix
        self.gram_matrix = torch.ones((pos_num, pos_num), dtype=torch.float32).cuda() * BIGNUM
        # Find the minimum allowed sample weight.Samples are discarded if their weights become lower
        self.minimum_sample_weight = opts['lr_init'] * np.power((1 - opts['lr_init']), (2 * pos_num))
        #  Initialize and allocate
        # self.prior_weights = torch.zeros((pos_num, 1), dtype=torch.float32).cuda()
        self.sample_weights = torch.zeros((pos_num, 1), dtype=torch.float32).cuda()
        self.num_training_samples = 0
        self.pos_num = pos_num
        self.samples = []
        self.pos_ious = []
        self.step_vis = 0
        self.last_added = -1
        if opts["debug"]:
            encoder = WAE64().cuda().eval()

            checkpoint = torch.load(opts["ckpt_path"])
            pretrain_dict = checkpoint['model_states']['net']
            model_dict = encoder.state_dict()
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict
                             and model_dict[k].size() == v.size()}
            encoder.load_state_dict(pretrain_dict)
            self.decoder = encoder.decoder
            del encoder

    def insert(self, new_train_samples, ious):

        if len(self.pos_features) < self.pos_num:
            self.pos_features.append(new_train_samples)
            self.samples.append(new_train_samples.mean(0))
            self.pos_ious.append(ious)
        else:
            self.distance_matrix = compute_distance_matrix(self.samples)
            self.gram_matrix = compute_gram_matrix(self.samples)
            self.update_sample_space_model(new_train_samples, ious)
            if opts["debug"]:
                a = torchvision.utils.make_grid(self.decoder(torch.stack(self.samples)), nrow=10)
                a = a.mul(255).byte()
                a = a.cpu().numpy().transpose((1, 2, 0))
                cv2.imshow("pos sample space", a)

            # writer.add_image("SampleSpace", a, self.step_vis)
        self.num_training_samples += 1

    def update_sample_space_model(self, new_train_samples, ious):
        # Set mean feature as the current features' representation
        new_train_sample = torch.sum(new_train_samples, 0) / len(new_train_samples)
        dist_vector = calc_distvector(new_train_sample.unsqueeze(0), torch.stack(self.samples))
        new_sample_id = -1
        # Find sample closest to the new sample
        new_sample_min_dist, closest_sample_to_new_sample = torch.min(dist_vector), torch.argmin(dist_vector)
        # Find the closest pair amongst existing samples
        existing_samples_min_dist, closest_existing_sample_pair = torch.min(self.distance_matrix.view(-1)), torch.argmin(self.distance_matrix.view(-1))
        closest_existing_sample1, closest_existing_sample2 = ind2sub(self.distance_matrix.shape, closest_existing_sample_pair)

        if torch.equal(closest_existing_sample1, closest_existing_sample2):
            os.error('Score matrix diagonal filled wrongly')

        if new_sample_min_dist < existing_samples_min_dist:
            new_sample_id = closest_sample_to_new_sample

            # Update distance matrix and the gram matrix
            self.update_distance_matrix(dist_vector, new_sample_id)

        if new_sample_id >= 0:
            # self.step_vis += 1

            self.pos_features[new_sample_id] = new_train_samples
            self.samples[new_sample_id] = new_train_samples.mean(0)
            self.pos_ious[new_sample_id] = ious

        if self.num_training_samples < self.pos_num:
            self.num_training_samples += 1
        self.last_added = new_sample_id

    def update_distance_matrix(self, dist_vector, new_id):
        if new_id >= 0:
            # Update distance matrix
            if self.distance_matrix[:, new_id].shape == dist_vector.t().shape:
                self.distance_matrix[:, new_id] = dist_vector.t()
                self.distance_matrix[new_id, :] = dist_vector
                self.distance_matrix[new_id, new_id] = BIGNUM
            else:
                self.distance_matrix[:, new_id] = dist_vector
                self.distance_matrix[new_id, :] = dist_vector
                self.distance_matrix[new_id, new_id] = BIGNUM
        elif new_id < 0:
            pass
            # The new sample is discared


class negative_samples_manger():
    def __init__(self, neg_num):
        # postive samples
        self.neg_features = []
        # Distance matrix stores the square of the euclidean distance between each pair of samples. Initialise it to inf
        self.distance_matrix = torch.ones((neg_num, neg_num), dtype=torch.float32).cuda() * BIGNUM
        # Kernel matrix, used to update distance matrix
        self.gram_matrix = torch.ones((neg_num, neg_num), dtype=torch.float32).cuda() * BIGNUM
        #  Initialize and allocate
        self.sample_weights = torch.zeros((neg_num, 1), dtype=torch.float32).cuda()
        self.num_training_samples = 0
        self.neg_num = neg_num
        self.samples = []
        self.neg_ious = []


    def insert(self, new_train_samples, ious):

        if len(self.neg_features) < self.neg_num:
            self.neg_features.append(new_train_samples)
            self.samples.append(new_train_samples.mean(0))
            self.neg_ious.append(ious)
        else:
            self.distance_matrix = compute_distance_matrix(self.samples)
            self.gram_matrix = compute_gram_matrix(self.samples)
            self.update_sample_space_model(new_train_samples, ious)
        self.num_training_samples += 1

    def update_sample_space_model(self, new_train_samples, ious):
        # Set mean feature as the current features' representation
        new_train_sample = torch.sum(new_train_samples, 0) / len(new_train_samples)
        dist_vector = calc_distvector(new_train_sample.unsqueeze(0), torch.stack(self.samples))

        new_sample_id = -1

        # Find sample closest to the new sample
        new_sample_min_dist, closest_sample_to_new_sample = torch.min(dist_vector), torch.argmin(dist_vector)
        # Find the closest pair amongst existing samples
        existing_samples_min_dist, closest_existing_sample_pair = torch.min(self.distance_matrix.view(-1)), torch.argmin(self.distance_matrix.view(-1))
        closest_existing_sample1, closest_existing_sample2 = ind2sub(self.distance_matrix.shape, closest_existing_sample_pair)

        if torch.equal(closest_existing_sample1, closest_existing_sample2):
            os.error('Score matrix diagonal filled wrongly')

        if new_sample_min_dist < existing_samples_min_dist:
            # Set the position of the merged sample
            new_sample_id = closest_sample_to_new_sample
            # Update distance matrix and the gram matrix
            self.update_distance_matrix(dist_vector, new_sample_id)

        if new_sample_id >= 0:
            # self.step_vis += 1
            self.neg_features[new_sample_id] = new_train_samples
            self.samples[new_sample_id] = new_train_samples.mean(0)
            self.neg_ious[new_sample_id] = ious
        if self.num_training_samples < self.neg_num:
            self.num_training_samples += 1
        self.last_added = new_sample_id

    def update_distance_matrix(self, dist_vector, new_id):
        if new_id >= 0:
            # Update distance matrix
            if self.distance_matrix[:, new_id].shape == dist_vector.t().shape:
                self.distance_matrix[:, new_id] = dist_vector.t()
                self.distance_matrix[new_id, :] = dist_vector
                self.distance_matrix[new_id, new_id] = BIGNUM
            else:
                self.distance_matrix[:, new_id] = dist_vector
                self.distance_matrix[new_id, :] = dist_vector
                self.distance_matrix[new_id, new_id] = BIGNUM
        elif new_id < 0:
            pass
            # The new sample is discared


def compute_distance_matrix(x):
    x = torch.stack(x)
    m, n = x.shape
    G = x.mm(x.t())
    H = G.diagonal().repeat((m,1))
    distance_matrix = H + H.t() - 2 * G
    
    for i in range(distance_matrix.shape[0]):
        distance_matrix[i][i] = BIGNUM

    return distance_matrix

def compute_gram_matrix(x):
    x = torch.stack(x)
    gram_matrix = x.mm(x.t())
    return gram_matrix

def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.int() / array_shape[1])
    cols = ind.int() % array_shape[1]
    return rows, cols

def calc_distvector(A, B):
    m = A.shape[0]
    n = B.shape[0]
    M = A.mm(B.t())
    H = A.pow(2).sum(dim=1).repeat(1, n)
    K = B.pow(2).sum(dim=1).repeat(m, 1)
    return torch.sqrt(-2 * M + H + K)


