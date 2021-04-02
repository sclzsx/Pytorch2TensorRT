from skimage.measure.simple_metrics import compare_psnr
from torch.autograd import Function
import torch
from math import sqrt as sqrt
from itertools import product as product
import numpy as np
import cv2
import random
import string

VOCroot = '/home/ipsg/code/sx/datasets'
COLORS_BGR = [(0, 0, 0), (0, 0, 0), (100, 101, 3), (154, 157, 252), (3, 155, 230), (78, 73, 209), (148, 137, 69)]
VOC_CLASSES = ['__background__', 'neg', 'robot', 'battery', 'brick', 'barrel', 'box']
assert len(COLORS_BGR) == len(VOC_CLASSES)
rgb_means = (104, 117, 123)

VOC_300 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [20, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
    'anchor_level_k': [9, 9, 9, 9, 9, 9]
}
VEHICLE_240 = {
    'feature_maps': [30, 15, 8, 4],
    'min_dim': 240,
    'steps': [8, 16, 30, 60],
    'min_sizes': [30, 80, 130, 180],
    'max_sizes': [80, 130, 180, 240],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
    'anchor_level_k': [3, 3, 3, 3]
}

input_size = 300
num_classes = len(VOC_CLASSES)


def cal_mean(list_, i=3):
    list_.sort()
    # return list_[-1]
    assert len(list_) > i
    list2 = list_[i:]
    return np.mean(list2)


def nms(dets, threshold):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    score = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    inds = score.argsort()[::-1]
    keep = []  # 保留的结果框集合
    while (inds.size > 0):
        top = inds[0]
        keep.append(top)  # 保留该类剩余box中得分最高的一个
        max_x1 = np.maximum(x1[top], x1[inds[1:]])
        max_y1 = np.maximum(y1[top], y1[inds[1:]])
        max_x2 = np.minimum(x2[top], x2[inds[1:]])
        max_y2 = np.minimum(y2[top], y2[inds[1:]])
        w = np.maximum(0.0, max_x2 - max_x1)
        h = np.maximum(0.0, max_y2 - max_y1)
        interarea = w * h
        IOU = interarea / (areas[top] + areas[inds[1:]] - interarea)
        order = np.where(IOU <= threshold)[0]
        inds = inds[order + 1]
    return keep


class PriorBox(object):
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class BaseTransform(object):
    def __init__(self, resize, rgb_means, swap=(2, 0, 1)):
        self.means = rgb_means
        self.resize = resize
        self.swap = swap

    def __call__(self, img):
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[0]
        img = cv2.resize(np.array(img), (self.resize, self.resize), interpolation=interp_method).astype(np.float32)
        img -= self.means
        img = img.transpose(self.swap)
        return torch.from_numpy(img)


def decode(loc, priors, variances):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


class Detect(Function):
    def __init__(self, num_classes, bkg_label, cfg):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.variance = cfg['variance']

    def forward(self, predictions, prior):
        loc, conf = predictions
        loc_data = loc.data
        conf_data = conf.data
        prior_data = prior.data
        num = loc_data.size(0)  # batch size
        self.num_priors = prior_data.size(0)
        self.boxes = torch.zeros(1, self.num_priors, 4)
        self.scores = torch.zeros(1, self.num_priors, self.num_classes)
        if loc_data.is_cuda:
            self.boxes = self.boxes.cuda()
            self.scores = self.scores.cuda()
        if num == 1:
            conf_preds = conf_data.unsqueeze(0)
        else:
            print('Only allow BZ = 1')
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i].clone()
            self.boxes[i] = decoded_boxes
            self.scores[i] = conf_scores
        return self.boxes, self.scores


def add_text_noise(img, stddev):
    img = img.copy()
    h, w, _ = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_for_cnt = np.zeros((h, w), np.uint8)
    occupancy = np.random.uniform(0, stddev)

    while True:
        n = random.randint(5, 10)
        random_str = ''.join([random.choice(string.ascii_letters + string.digits) for i in range(n)])
        font_scale = np.random.uniform(0.5, 1)
        thickness = random.randint(1, 3)
        (fw, fh), baseline = cv2.getTextSize(random_str, font, font_scale, thickness)
        x = random.randint(0, max(0, w - 1 - fw))
        y = random.randint(fh, h - 1 - baseline)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.putText(img, random_str, (x, y), font, font_scale, color, thickness)
        cv2.putText(img_for_cnt, random_str, (x, y), font, font_scale, 255, thickness)

        if (img_for_cnt > 0).sum() > h * w * occupancy / 100:
            break
    return img


def add_gaussian_noise(img, stddev):
    noise = np.random.randn(*img.shape) * stddev
    noise_img = img + noise
    noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
    return noise_img


def add_impulse_noise(img, stddev):
    occupancy = np.random.uniform(0, stddev)
    mask = np.random.binomial(size=img.shape, n=1, p=occupancy / 100)
    noise = np.random.randint(256, size=img.shape)
    img = img * (1 - mask) + noise * mask
    return img.astype(np.uint8)


def add_multi_noise(img, stddev):
    img = img.copy()
    g = add_gaussian_noise(img, stddev)
    t = add_text_noise(g, stddev)
    i = add_impulse_noise(t, stddev)
    return i


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])
