import os
import sys

import cv2
from core.bbreg.bbreg_nn_klloss import BBRegNN
from net.loss.ghmseloss import GHMSE_Loss
from net.loss.mmd import mmd
from tracking.options import opts
import numpy as np
sys.path.append(os.path.abspath('../'))
from torch.autograd import Variable
from core.trackers.tracker_base import tracker
from net.classifier.fc_hf import FC_HF
from net.classifier.HessianFree import HessianFree
from net.classifier.Lookahead import Lookahead
from core.samples.samples import Memory
from net.wae64 import WAE64
from core.samples.sample_generator import SampleGenerator, gen_bboxes
import torch
from core.utils import shuffleTensor, sample_bbox, overlap_ratio

class DCFT_FCHF(tracker):
    def __init__(self, frame=None, init_bbox=None, gt=None):
        tracker.__init__(self, frame, init_bbox)
        # Generate sequence config
        self.first_frame = frame
        self.init_bbox = init_bbox
        self.current_bbox = init_bbox
        self.img_sz = (frame.shape[1], frame.shape[0])
        self.gt = gt

        #########   Encoder    ##################
        self.encoder = WAE64().cuda().eval()
        # classifer and it's delayed updated brothers
        self.classifier = FC_HF(in_dim=opts['in_dim'], l_dim=opts['l_dim']).cuda()
        # checkpt_clc = torch.load("../net/train/pretrain_classifier/model_best.pth.tar")
        # self.classifier.load_state_dict(checkpt_clc['state_dict'])

        # self.classifier_delay = FC_HF(in_dim=opts['in_dim'], l_dim=opts['l_dim']).cuda()
        self.update_step = 0

        # self.criterion = torch.nn.MSELoss()
        # self.criterion = GHMR_Loss(bins=30, alpha=0, mu=0.02)
        self.criterion = GHMSE_Loss(bins=30, alpha=0, mu=0.02)

        # self.optimizer = Adam(self.classifier.parameters())
        base_opt = torch.optim.Adam(self.classifier.parameters(), lr=1e-3, betas=(0.9, 0.999))  # Any optimizer
        self.optimizer = Lookahead(base_opt, k=5, alpha=0.5)  # Initialize Lookahead

        # defined to generate the candidate damples
        self.sample_generator = SampleGenerator('gaussian', self.img_sz, opts['trans'], opts['scale'])

        ## Bounding box regression
        # self.bbreg = BBRegressor(self.img_sz)
        self.bbreg = BBRegNN(self.img_sz)
        # how many frames did we track
        self.step = 1

        # if > 3, then set the opts['long_interval'] = 15
        self.fail_num = 0
        ## Sampler
        self.memory = Memory(extractor=self.encoder)

        ## Trace prediction at failure

        init_cx, init_cy = int(self.init_bbox[0] + self.init_bbox[2] / 2), int(self.init_bbox[1] + self.init_bbox[3] / 2)
        self.predictedCoords = np.array((init_cx, init_cy), np.int)

    # hm
    def train_cls(self, pos_feats, neg_feats, params, iter=opts['init_freq']):
        pos_ious = np.concatenate(params[0])
        neg_ious = np.concatenate(params[1])

        self.classifier.train()
        batch_pos = opts['batch_pos']
        batch_neg = opts['batch_neg']
        batch_test = opts['batch_test']
        batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

        pos_idx = np.random.permutation(pos_feats.size(0))
        neg_idx = np.random.permutation(neg_feats.size(0))
        while (len(pos_idx) < batch_pos * iter):
            pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
        while (len(neg_idx) < batch_neg_cand * iter):
            neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
        pos_pointer = 0
        neg_pointer = 0

        for i in range(iter):
            self.optimizer.zero_grad()
            # select pos idx
            pos_next = pos_pointer + batch_pos
            pos_cur_idx = pos_idx[pos_pointer:pos_next]
            pos_cur_idx = pos_feats.new(pos_cur_idx).long()
            pos_pointer = pos_next

            # select neg idx
            neg_next = neg_pointer + batch_neg_cand
            neg_cur_idx = neg_idx[neg_pointer:neg_next]
            neg_cur_idx = neg_feats.new(neg_cur_idx).long()
            neg_pointer = neg_next

            # create batch
            batch_pos_feats = pos_feats[pos_cur_idx]
            batch_pos_ious = pos_ious[pos_cur_idx.cpu().numpy()]
            batch_neg_feats = neg_feats[neg_cur_idx]
            batch_neg_ious = neg_ious[neg_cur_idx.cpu().numpy()]

            # hard negative mining
            if batch_neg_cand > batch_neg:
                self.classifier.eval()
                for start in range(0, batch_neg_cand, batch_test):
                    end = min(start + batch_test, batch_neg_cand)
                    with torch.no_grad():
                        score, _ = self.classifier.forward(batch_neg_feats[start:end])
                    if start == 0:
                        neg_cand_score = score.detach()[:, 1].clone()
                    else:
                        neg_cand_score = torch.cat((neg_cand_score, score.detach()[:, 1].clone()), 0)

                _, top_idx = neg_cand_score.topk(batch_neg)
                batch_neg_feats = batch_neg_feats[top_idx]
                batch_neg_ious = batch_neg_ious[top_idx.cpu().numpy()]
            self.classifier.train()

            code, target = shuffleTensor(batch_pos_feats, batch_neg_feats, [batch_pos_ious, batch_neg_ious])

            # pos_old_code = code[target[:, 1].gt(target[:, 0]), :]
            # neg_old_code = code[target[:, 0].gt(target[:, 1]), :]

            score, new_code = self.classifier(code)
            # calc classifier loss
            classifier_loss = self.criterion(score, target)

            pos_new_code = new_code[target[:, 1].gt(target[:, 0]), :]
            neg_new_code = new_code[target[:, 0].gt(target[:, 1]), :]

            # calc two distribution's mmd loss
            # pos_code_loss = torch.abs(mmd(pos_new_code, pos_old_code, z_var=2))
            # neg_code_loss = torch.abs(mmd(neg_new_code, neg_old_code, z_var=2))
            # Dimension alignment
            idx = np.random.permutation(pos_new_code.size(0))
            # pn_old_code_loss = torch.abs(mmd(batch_pos_feats, batch_neg_feats[idx], z_var=2))
            pn_new_code_loss = torch.abs(mmd(pos_new_code, neg_new_code[idx], z_var=2))
            # pn_old_code_eu = torch.sqrt(
            #     torch.sum((torch.cat([batch_pos_feats, batch_pos_feats, batch_pos_feats]) - batch_neg_feats).pow(2)))
            # pn_new_code_eu = torch.sqrt(
            #     torch.sum((torch.cat([pos_new_code, pos_new_code, pos_new_code]) - neg_new_code).pow(2)))

            # m = pn_new_code_loss - pn_old_code_loss
            # m > 0 : reward
            # m < 0: punish     torch.log(1/pn_new_code_loss) +
            # loss = classifier_loss / (1 + pn_new_code_loss) + (pos_code_loss + neg_code_loss) / (0.1 + torch.exp(m + 2))
            loss = classifier_loss - torch.log(pn_new_code_loss)
            # loss = classifier_loss - torch.log(pn_new_code_eu)

            # writer.add_scalar("{}/distance新欧几里得".format(opts["seq_name"]), pn_new_code_eu, (self.step-1)*iter + i)
            # writer.add_scalar("{}/distance旧欧几里得".format(opts["seq_name"]), pn_old_code_eu, (self.step-1)*iter + i)
            # print((self.step-1)*iter + i)
            #
            # writer.add_scalar("{}/distance新MMD".format(opts["seq_name"]), pn_new_code_loss, self.step)
            # writer.add_scalar("{}/distance旧MMD".format(opts["seq_name"]), pn_old_code_loss, self.step)

            self.optimizer.zero_grad()
            loss.backward()
            if 'grad_clip' in opts:
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), opts['grad_clip'])
            self.optimizer.step()

        # delayed update classifier_delay
        # if self.update_step % opts['delay_freq'] == 0:
        #     self.classifier_delay.load_state_dict(deepcopy(self.classifier.state_dict()))
        self.update_step += 1

    # Hard negative and hard negtive positive
    def train_cls_ohem(self, pos_feats, neg_feats, params, iter=opts['init_freq']):
        pos_ious = np.concatenate(params[0])
        neg_ious = np.concatenate(params[1])

        self.classifier.train()
        batch_pos = opts['batch_pos']
        batch_neg = opts['batch_neg']
        batch_test_neg = opts['batch_test_neg']
        batch_test_pos = opts['batch_test_pos']

        batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)
        batch_pos_cand = max(opts['batch_pos_cand'], batch_pos)

        pos_idx = np.random.permutation(pos_feats.size(0))
        neg_idx = np.random.permutation(neg_feats.size(0))
        while (len(pos_idx) < batch_pos_cand * iter):
            pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
        while (len(neg_idx) < batch_neg_cand * iter):
            neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
        pos_pointer = 0
        neg_pointer = 0

        for i in range(iter):
            self.optimizer.zero_grad()
            # select pos idx
            pos_next = pos_pointer + batch_pos_cand  # batch_pos
            pos_cur_idx = pos_idx[pos_pointer:pos_next]
            pos_cur_idx = pos_feats.new(pos_cur_idx).long()
            pos_pointer = pos_next

            # select neg idx
            neg_next = neg_pointer + batch_neg_cand
            neg_cur_idx = neg_idx[neg_pointer:neg_next]
            neg_cur_idx = neg_feats.new(neg_cur_idx).long()
            neg_pointer = neg_next

            # create batch
            batch_pos_feats = pos_feats[pos_cur_idx]
            batch_pos_ious = pos_ious[pos_cur_idx.cpu().numpy()]
            batch_neg_feats = neg_feats[neg_cur_idx]
            batch_neg_ious = neg_ious[neg_cur_idx.cpu().numpy()]

            # hard negative samples
            if batch_neg_cand > batch_neg:
                self.classifier.eval()
                for start in range(0, batch_neg_cand, batch_test_neg):
                    end = min(start + batch_test_neg, batch_neg_cand)
                    with torch.no_grad():
                        score, _ = self.classifier.forward(batch_neg_feats[start:end])
                    if start == 0:
                        neg_cand_score = score.detach()[:, 1].clone()
                    else:
                        neg_cand_score = torch.cat((neg_cand_score, score.detach()[:, 1].clone()), 0)

                _, top_idx = neg_cand_score.topk(batch_neg)
                batch_neg_feats = batch_neg_feats[top_idx]
                batch_neg_ious = batch_neg_ious[top_idx.cpu().numpy()]

            # hard positive samples
            if batch_pos_cand > batch_pos:
                self.classifier.eval()
                for start in range(0, batch_pos_cand, batch_test_pos):
                    end = min(start + batch_test_pos, batch_pos_cand)
                    with torch.no_grad():
                        score, _ = self.classifier.forward(batch_pos_feats[start:end])
                    if start == 0:
                        pos_cand_score = score.detach()[:, 1].clone()
                    else:
                        pos_cand_score = torch.cat((pos_cand_score, score.detach()[:, 1].clone()), 0)

                _, top_idx = pos_cand_score.topk(batch_pos, largest=False)
                batch_pos_feats = batch_pos_feats[top_idx]
                batch_pos_ious = batch_pos_ious[top_idx.cpu().numpy()]

            self.classifier.train()

            code, target = shuffleTensor(batch_pos_feats, batch_neg_feats, [batch_pos_ious, batch_neg_ious])

            pos_old_code = code[target[:, 1].gt(target[:, 0]), :]
            neg_old_code = code[target[:, 0].gt(target[:, 1]), :]

            score, new_code = self.classifier(code)
            # calc classifier loss
            classifier_loss = self.criterion(score, target)

            pos_new_code = new_code[target[:, 1].gt(target[:, 0]), :]
            neg_new_code = new_code[target[:, 0].gt(target[:, 1]), :]

            # calc two distribution's mmd loss
            pos_code_loss = torch.abs(mmd(pos_new_code, pos_old_code, z_var=2))
            neg_code_loss = torch.abs(mmd(neg_new_code, neg_old_code, z_var=2))
            # Dimension alignment
            idx = np.random.permutation(pos_new_code.size(0))
            pn_old_code_loss = torch.abs(mmd(batch_pos_feats, batch_neg_feats[idx], z_var=2))
            pn_new_code_loss = torch.abs(mmd(pos_new_code, neg_new_code[idx], z_var=2))
            # pn_old_code_eu = torch.sqrt(
            #     torch.sum((torch.cat([batch_pos_feats, batch_pos_feats, batch_pos_feats]) - batch_neg_feats).pow(2)))
            # pn_new_code_eu = torch.sqrt(
            #     torch.sum((torch.cat([pos_new_code, pos_new_code, pos_new_code]) - neg_new_code).pow(2)))

            m = pn_new_code_loss - pn_old_code_loss
            # m > 0 : reward
            # m < 0: punish     torch.log(1/pn_new_code_loss) +
            # loss = classifier_loss / (1 + pn_new_code_loss) + (pos_code_loss + neg_code_loss) / (0.1 + torch.exp(m + 2))
            loss = classifier_loss - torch.log(pn_new_code_loss)
            # loss = classifier_loss - torch.log(pn_new_code_eu)

            self.optimizer.zero_grad()
            loss.backward()
            if 'grad_clip' in opts:
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), opts['grad_clip'])
            self.optimizer.step()

        # delayed update classifier_delay
        # if self.update_step % opts['delay_freq'] == 0:
        #     self.classifier_delay.load_state_dict(deepcopy(self.classifier.state_dict()))
        self.update_step += 1

    def init_bbreg(self):
        # Train bbox regressor
        # bbreg_rects = gen_bboxes(
        #     SampleGenerator('uniform', self.img_sz, opts['trans_bbreg'], opts['scale_bbreg'], opts['aspect_bbreg']),
        #     self.init_bbox, opts['n_bbreg'], opts['overlap_bbreg'])
        # bbreg_feats = self.memory.extract_feats(self.first_frame, bbreg_rects)

        self.bbreg = BBRegNN(self.img_sz)
        # self.bbreg.train(bbreg_feats, bbreg_rects, self.init_bbox)

        # self.bbreg = BBRegressor(self.img_sz)
        # self.bbreg.train(bbreg_feats, bbreg_rects, self.init_bbox)

    def stage_two(self, frame=None, candidate=None):
        global bbreg_bbox
        old_bbox = self.current_bbox
        old_bbreg_bbox = self.bbreg_bbox

        # Candidate locations & it's scores
        rects = gen_bboxes(self.sample_generator, old_bbox, opts['n_samples'])

        feats = self.memory.extract_feats(frame, rects)

        self.classifier.eval()

        sample_scores1, _ = self.classifier(feats)  # last frame classifier
        sample_scores1 = sample_scores1[:, 1].detach()

        sample_scores = sample_scores1  # * opts['score_ratio'] + sample_scores2 * (1 - opts['score_ratio'])

        top_scores, top_idx = sample_scores.topk(5)
        top_idx = top_idx.cpu().numpy()
        target_score = top_scores.mean()
        target_bbox = rects[top_idx].mean(axis=0)

        success = target_score > opts['success_thr']

        # Expand search area at failure
        if success:
            self.sample_generator.set_trans(opts['trans'])
        else:
            self.sample_generator.expand_trans(opts['trans_limit'])
        # Bbox regression
        if success:
            self.fail_num = 0
            bbreg_samples = rects[top_idx]
            bbreg_feats = self.memory.extract_feats(frame, bbreg_samples)
            bbreg_samples = self.bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            self.fail_num += 1

            pos_feats, neg_feats, pos_ious, neg_ious = self.memory.get_features(type='supdate',
                                                                                    samples=opts['n_frames_update'])
            self.train_cls(pos_feats, neg_feats, params=[pos_ious, neg_ious], iter=opts['update_freq'])
            # self.train_cls_ohem(pos_feats, neg_feats, params=[pos_ious, neg_ious], iter=opts['update_freq'])

            # if not success, then search in a larger range
            rects = np.random.permutation(np.concatenate([
                gen_bboxes(self.sample_generator, old_bbreg_bbox, opts['n_samples'] * 3),
                gen_bboxes(SampleGenerator('uniform', self.img_sz, opts['trans_neg_init'], opts['scale_neg_init']),
                           old_bbox, opts['n_samples'], opts['overlap_neg_init']),
                gen_bboxes(SampleGenerator('uniform', self.img_sz, opts['trans_neg_init'], opts['scale_neg_init']),
                           old_bbox, opts['n_samples'], opts['overlap_neg_init']),

            ]))
            rects = np.random.permutation(rects)

            feats = self.memory.extract_feats(frame, rects)

            sample_scores1, _ = self.classifier(feats)
            sample_scores1 = sample_scores1[:, 1].detach()

            sample_scores = sample_scores1  # * opts['score_ratio'] + sample_scores2 * (1 - opts['score_ratio'])

            top_scores, top_idx = sample_scores.topk(5)
            top_idx = top_idx.cpu().numpy()
            target_score = top_scores.mean()

            success = target_score > opts['success_thr'] # and sample_scores.mean().item() > 0.1
            if success:
                target_bbox = rects[top_idx].mean(axis=0)
                bbreg_samples = rects[top_idx]
                bbreg_feats = self.memory.extract_feats(frame, rects)
                bbreg_samples = self.bbreg.predict(bbreg_feats[top_idx], bbreg_samples)
                bbreg_bbox = bbreg_samples.mean(axis=0)

        # Still not Success, use kalman filter's results
        if not success:
            target_bbox = old_bbox
            bbreg_bbox = old_bbreg_bbox
            return target_bbox, bbreg_bbox, target_score
        # Data collect
        if success:
            self.memory.add_features(frame, target_bbox)

        if self.step % opts['long_interval'] == 0:
            # Get all features to update model
            pos_data, neg_data, pos_ious, neg_ious = self.memory.get_features()
            self.train_cls(pos_data, neg_data, params=[pos_ious, neg_ious], iter=opts['init_freq'])
            # self.train_cls_ohem(pos_data, neg_data, params=[pos_ious, neg_ious], iter=opts['init_freq'])

        return target_bbox, bbreg_bbox, target_score

    def init(self):
        # init frame and boxes
        self.current_bbox = self.init_bbox
        self.bbreg_bbox = self.init_bbox
        # weather small target or not
        if (self.init_bbox[2] * self.init_bbox[3]) / (self.img_sz[0] * self.img_sz[1]) < 0.001:
            opts['scale'] = 1
            opts['scale_pos'] = 1.05
            opts['scale_neg_init'] = 1.2
            opts['scale_neg'] = 1.1
            opts['scale_bbreg'] = 1.1
            opts['success_thr'] = 0.5

        # init memory manager
        self.memory.init(self.first_frame, self.init_bbox)
        # init
        self.init_bbreg()

        # get all features in the first frame
        pos_feats, neg_feats, pos_ious, neg_ious = self.memory.get_features()
        self.train_cls(pos_feats, neg_feats, params=[pos_ious, neg_ious])
        # self.train_cls_ohem(pos_feats, neg_feats, params=[pos_ious, neg_ious])

        self.memory.del_features(n_pos=opts['n_pos_update'], n_neg=opts['n_neg_update'])
        self.old_frame = self.first_frame

    def update(self, frame):
        bbox, bbreg_bbox, score = self.stage_two(frame=frame, candidate=None)
        self.old_frame = frame
        self.current_bbox = bbox
        self.bbreg_bbox = bbreg_bbox
        self.step += 1
        if self.fail_num == 0:
            cx, cy = bbreg_bbox[0] + bbreg_bbox[2] / 2, bbreg_bbox[1] + bbreg_bbox[3] / 2
        return bbox, bbreg_bbox, score

    def asymmetricKL(self, P, Q):
        return torch.sum(P * torch.log(P / Q))  # calculate the kl divergence between P and Q

    def symmetricalKL(self, P, Q):
        return (self.asymmetricKL(P, Q) + self.asymmetricKL(Q, P)) / 2.00

    def stridebox(self, rect, new_center):
        # TODO
        w, h = rect[2:]
        new_x, new_y = new_center[0] - rect[2] / 2., new_center[1] - rect[3] / 2.,
        new_rect = np.array([new_x, new_y, w, h])

        old_cx, old_cy = int(rect[0] + rect[2] / 2), int(rect[1] + rect[3] / 2)
        frame = self.old_frame.copy()
        cv2.circle(frame, (old_cx, old_cy), 20, [255, 0, 0], 2, 8)
        cv2.line(frame, (old_cx - 16, old_cy - 15),
                 (old_cx - 50, old_cy - 30), [255, 0, 0], 2, 8)
        cv2.putText(frame, "Original",
                    (int(old_cx - 120), int(old_cy - 30)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, [255, 0, 0])

        new_cx, new_cy = int(self.predictedCoords[0]), int(self.predictedCoords[1])

        cv2.circle(frame, (new_cx, new_cy), 20, [0, 0, 255], 2, 8)
        cv2.line(frame, (new_cx + 16, new_cy - 15),
                 (new_cx + 50, new_cy - 30), [0, 0, 255], 2, 8)
        cv2.putText(frame, "Predicted",
                    (int(new_cx + 50), int(new_cy - 30)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, [0, 0, 255])
        cv2.imshow('Kalman', frame)
        cv2.waitKey(100)

        return new_rect
