from core.bbreg.bbreg_model import BBregNet
from core.bbreg.utils.boxes import xywh_to_xyxy, bbox_transform_inv, bbox_transform, xyxy_to_xywh
from core.utils import *
from net.loss.giouloss import giou


class BBRegNN():
    def __init__(self, img_size, alpha=1000, overlap=[0.6, 1], scale=[1, 2]):
        self.img_size = img_size
        self.alpha = alpha
        self.overlap_range = overlap
        self.scale_range = scale
        self.model = BBregNet().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        chk_pt = "../net/weights/checkpoint.pth.tar"
        checkpoint = torch.load(chk_pt)
        self.model.load_state_dict(checkpoint['state_dict'])

    def train(self, X, bbox, gt, batch_sz=32, epoch=50):
        self.model.train()
        bbox = np.copy(bbox)
        gt = np.copy(gt)

        if gt.ndim == 1:
            gt = gt[None, :]

        gt = xywh_to_xyxy(gt)
        boxes = xywh_to_xyxy(bbox)

        target = bbox_transform_inv(boxes, gt)
        target = torch.from_numpy(target.astype(np.float32)).cuda()

        num_iter = len(target) // batch_sz
        for epo in range(epoch):
            ptr = 0
            for i in range(num_iter):
                if 0 < len(target) - ptr < batch_sz:
                    x = X[ptr:, :]
                    y = target[ptr:, :]
                    b = boxes[ptr:, :]
                x = X[ptr:(ptr + batch_sz), :]
                y = target[ptr:(ptr + batch_sz), :]
                b = boxes[ptr:(ptr + batch_sz), :]

                ptr += batch_sz

                dealta, std = self.model(x)
                alpha = 1 / std.pow(2)
                klloss = torch.exp(-alpha) * (torch.abs(y - dealta) - 0.5) + 0.5 * alpha
                klloss = klloss.sum() / (klloss.shape[0] * klloss.shape[1])

                b = torch.from_numpy(bbox_transform(b, dealta.cpu().detach().numpy())).cuda().float()
                g = torch.from_numpy(gt).cuda().float()
                iouloss = giou(b, g).sum() / batch_sz

                loss = klloss #+ iouloss
                # print(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, X, bbox):
        bbox_ = np.copy(bbox)
        bbox_ = xywh_to_xyxy(bbox_)
        self.model.eval()
        dealta, std = self.model(X)

        bbox = bbox_transform(bbox_, dealta.cpu().detach().numpy(), )
        bbox = xyxy_to_xywh(bbox)
        return bbox

    def get_examples(self, bbox, gt):
        bbox[:, 2:] = bbox[:, :2] + bbox[:, 2:]
        gt[:, 2:] = gt[:, :2] + gt[:, 2:]

        dst_x1y1 = (gt[:, :2] - bbox[:, :2]) / bbox[:, 2:]
        dst_x2y2 = (gt[:, 2:] - bbox[:, 2:]) / bbox[:, 2:]

        Y = np.concatenate((dst_x1y1, dst_x2y2), axis=1)
        return Y

    def xywh2xyxy(self, rects):
        w, h = rects[:, 2:]
        x1, y1 = rects[:, :2]
        x2, y2 = x1 + w, y1 + h
        return np.array(x1, y1, x2, y2)
