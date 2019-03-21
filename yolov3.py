import argparse

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import time
import copy
from pathlib import Path

import numpy as np

from classifier import Classifier
from collections import defaultdict
from classifier_utils import setup_default_args, parse_model_cfg
from pruning.masked_conv_2d import MaskedConv2d
from pruning.masked_linear import MaskedLinear
from pruning.masked_sequential import MaskedSequential
from pruning.methods import weight_prune, prune_rate

from repos.yolov3.utils.datasets import LoadImagesAndLabels

class MaskedModuleList(nn.ModuleList):
    def __init__(self):
        super(MaskedModuleList, self).__init__()
    
    def set_mask(self, masks):
        for module, mask in zip(self.children(), masks[0]):
            module.set_mask(mask)


def return_torch_unique_index(u, uv):
    n = uv.shape[1]  # number of columns
    first_unique = torch.zeros(n, device=u.device).long()
    for j in range(n):
        first_unique[j] = (uv[:, j:j + 1] == u).all(0).nonzero()[0]

    return first_unique

def build_targets(target, anchor_vec, nA, nC, nG):
    """
    returns nT, nCorrect, tx, ty, tw, th, tconf, tcls
    """
    nB = len(target)  # number of images in batch

    txy = torch.zeros(nB, nA, nG, nG, 2)  # batch size, anchors, grid size
    twh = torch.zeros(nB, nA, nG, nG, 2)
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)  # nC = number of classes

    for b in range(nB):
        t = target[b]
        nTb = len(t)  # number of targets
        if nTb == 0:
            continue

        gxy, gwh = t[:, 1:3] * nG, t[:, 3:5] * nG

        # Get grid box indices and prevent overflows (i.e. 13.01 on 13 anchors)
        gi, gj = torch.clamp(gxy.long(), min=0, max=nG - 1).t()

        # iou of targets-anchors (using wh only)
        box1 = gwh
        box2 = anchor_vec.unsqueeze(1)

        inter_area = torch.min(box1, box2).prod(2)
        iou = inter_area / (box1.prod(1) + box2.prod(2) - inter_area + 1e-16)

        # Select best iou_pred and anchor
        iou_best, a = iou.max(0)  # best anchor [0-2] for each target

        # Select best unique target-anchor combinations
        if nTb > 1:
            iou_order = torch.argsort(-iou_best)  # best to worst

            # Unique anchor selection
            u = torch.stack((gi, gj, a), 0)[:, iou_order]
            # _, first_unique = np.unique(u, axis=1, return_index=True)  # first unique indices
            first_unique = return_torch_unique_index(u, torch.unique(u, dim=1))  # torch alternative

            i = iou_order[first_unique]
            # best anchor must share significant commonality (iou) with target
            i = i[iou_best[i] > 0.10]  # TODO: examine arbitrary threshold
            if len(i) == 0:
                continue

            a, gj, gi, t = a[i], gj[i], gi[i], t[i]
            if len(t.shape) == 1:
                t = t.view(1, 5)
        else:
            if iou_best < 0.10:
                continue

        tc, gxy, gwh = t[:, 0].long(), t[:, 1:3] * nG, t[:, 3:5] * nG

        # XY coordinates
        txy[b, a, gj, gi] = gxy - gxy.floor()

        # Width and height
        twh[b, a, gj, gi] = torch.log(gwh / anchor_vec[a])  # yolo method
        # twh[b, a, gj, gi] = torch.sqrt(gwh / anchor_vec[a]) / 2 # power method

        # One-hot encoding of label
        tcls[b, a, gj, gi, tc] = 1
        tconf[b, a, gj, gi] = 1

    return txy, twh, tconf, tcls



def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, 'r')
    names = fp.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = MaskedModuleList()
    yolo_layer_count = 0
    for i, module_def in enumerate(module_defs):
        modules = MaskedSequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, MaskedConv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                modules.add_module('_debug_padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module('maxpool_%d' % i, maxpool)

        elif module_def['type'] == 'upsample':
            # upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')  # WARNING: deprecated
            upsample = Upsample(scale_factor=int(module_def['stride']))
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())

        elif module_def['type'] == 'yolo':
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
            # Extract anchors
            anchors = [float(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            nC = int(module_def['classes'])  # number of classes
            img_size = int(hyperparams['height'])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, nC, img_size, yolo_layer_count, cfg=hyperparams['cfg'])
            modules.add_module('yolo_%d' % i, yolo_layer)
            yolo_layer_count += 1

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nC, img_size, yolo_layer, cfg, device='cuda:1'):
        super(YOLOLayer, self).__init__()

        nA = len(anchors)
        self.anchors = torch.FloatTensor(anchors)
        self.nA = nA  # number of anchors (3)
        self.nC = nC  # number of classes (80)
        self.img_size = 0
        self.device = device
        # self.coco_class_weights = coco_class_weights()


    def forward(self, p, img_size, targets=None, var=None):
        bs, nG = p.shape[0], p.shape[-1]

        if self.img_size != img_size:
            create_grids(self, img_size, nG)

            if p.is_cuda:
                self.grid_xy = self.grid_xy.cuda(self.device)
                self.anchor_wh = self.anchor_wh.cuda(self.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 80)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.nA, self.nC + 5, nG, nG).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        # xy, width and height
        xy = torch.sigmoid(p[..., 0:2])
        wh = p[..., 2:4]  # wh (yolo method)
        # wh = torch.sigmoid(p[..., 2:4])  # wh (power method)

        # Training
        if targets is not None:
            MSELoss = nn.MSELoss()
            BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
            CrossEntropyLoss = nn.CrossEntropyLoss()

            # Get outputs
            p_conf = p[..., 4]  # Conf
            p_cls = p[..., 5:]  # Class

            txy, twh, mask, tcls = build_targets(targets, self.anchor_vec, self.nA, self.nC, nG)

            tcls = tcls[mask]
            if p.is_cuda:
                txy, twh, mask, tcls = txy.cuda(self.device), twh.cuda(self.device), mask.cuda(self.device), tcls.cuda(self.device)

            # Compute losses
            nT = sum([len(x) for x in targets])  # number of targets
            nM = mask.sum().float()  # number of anchors (assigned to targets)
            k = 1  # nM / bs
            if nM > 0:
                lxy = k * MSELoss(xy[mask], txy[mask])
                lwh = k * MSELoss(wh[mask], twh[mask])

                lcls = (k / 4) * CrossEntropyLoss(p_cls[mask], torch.argmax(tcls, 1))
                # lcls = (k * 10) * BCEWithLogitsLoss(p_cls[mask], tcls.float())
            else:
                FT = torch.cuda.FloatTensor
                lxy, lwh, lcls, lconf = FT([0]).cuda(self.device), FT([0]).cuda(self.device), FT([0]).cuda(self.device), FT([0]).cuda(self.device)

            lconf = (k * 64) * BCEWithLogitsLoss(p_conf, mask.float())

            # Sum loss components
            loss = lxy + lwh + lconf + lcls

            return loss, loss.item(), lxy.item(), lwh.item(), lconf.item(), lcls.item(), nT

        else:
            p[..., 0:2] = xy + self.grid_xy  # xy
            p[..., 2:4] = torch.exp(wh) * self.anchor_wh  # wh yolo method
            # p[..., 2:4] = ((wh * 2) ** 2) * self.anchor_wh  # wh power method
            p[..., 4] = torch.sigmoid(p[..., 4])  # p_conf
            p[..., :4] *= self.stride

            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return p.view(bs, -1, 5 + self.nC)


class MaskedDarknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, cfg_path, img_size=416):
        super(MaskedDarknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg_path)
        self.module_defs[0]['cfg'] = cfg_path
        self.module_defs[0]['height'] = img_size
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.loss_names = ['loss', 'xy', 'wh', 'conf', 'cls', 'nT']
        self.losses = []

    def forward(self, x, targets=None, var=0):
        self.losses = defaultdict(float)
        is_training = targets is not None
        img_size = x.shape[-1]
        layer_outputs = []
        output = []

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = module_def['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif mtype == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif mtype == 'yolo':
                if is_training:  # get loss
                    x, *losses = module[0](x, img_size, targets, var)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                else:  # get detections
                    x = module[0](x, img_size)
                output.append(x)
            layer_outputs.append(x)

        if is_training:
            self.losses['nT'] /= 3

        return sum(output) if is_training else torch.cat(output, 1)
    
    def set_mask(self, mask):
        self.module_list.set_mask(mask)

def get_yolo_layers(model):
    a = [module_def['type'] == 'yolo' for module_def in model.module_defs]
    return [i for i, x in enumerate(a) if x]  # [82, 94, 106] for yolov3


def create_grids(self, img_size, nG):
    self.stride = img_size / nG

    # build xy offsets
    grid_x = torch.arange(nG).repeat((nG, 1)).view((1, 1, nG, nG)).float()
    grid_y = grid_x.permute(0, 1, 3, 2)
    self.grid_xy = torch.stack((grid_x, grid_y), 4)

    # build wh gains
    self.anchor_vec = self.anchors / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.nA, 1, 1, 2)


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'
    # cutoff: save layers between 0 and cutoff (if cutoff = -1 all are saved)
    weights_file = weights.split(os.sep)[-1]

    # Try to download weights if not available locally
    if not os.path.isfile(weights):
        try:
            os.system('wget https://pjreddie.com/media/files/' + weights_file + ' -O ' + weights)
        except IOError:
            print(weights + ' not found')

    # Establish cutoffs
    if weights_file == 'darknet53.conv.74':
        cutoff = 75
    elif weights_file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Open the weights file
    fp = open(weights, 'rb')
    header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

    # Needed to write header when saving weights
    self.header_info = header

    self.seen = header[3]  # number of images seen during training
    weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
    fp.close()

    ptr = 0
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            if module_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w


"""
    @:param path    - path of the new weights file
    @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
"""


def save_weights(self, path, cutoff=-1):
    fp = open(path, 'wb')
    self.header_info[3] = self.seen  # number of images seen during training
    self.header_info.tofile(fp)

    # Iterate through layers
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            # If batch norm, load bn first
            if module_def['batch_normalize']:
                bn_layer = module[1]
                bn_layer.bias.data.cpu().numpy().tofile(fp)
                bn_layer.weight.data.cpu().numpy().tofile(fp)
                bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                bn_layer.running_var.data.cpu().numpy().tofile(fp)
            # Load conv bias
            else:
                conv_layer.bias.data.cpu().numpy().tofile(fp)
            # Load conv weights
            conv_layer.weight.data.cpu().numpy().tofile(fp)

    fp.close()

def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    output = [None for _ in range(len(prediction))]
    for image_i, pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        # Get score and class with highest confidence

        # cross-class NMS (experimental)
        cross_class_nms = False
        if cross_class_nms:
            a = pred.clone()
            _, indices = torch.sort(-a[:, 4], 0)  # sort best to worst
            a = a[indices]
            radius = 30  # area to search for cross-class ious
            for i in range(len(a)):
                if i >= len(a) - 1:
                    break

                close = (torch.abs(a[i, 0] - a[i + 1:, 0]) < radius) & (torch.abs(a[i, 1] - a[i + 1:, 1]) < radius)
                close = close.nonzero()

                if len(close) > 0:
                    close = close + i + 1
                    iou = bbox_iou(a[i:i + 1, :4], a[close.squeeze(), :4].reshape(-1, 4), x1y1x2y2=False)
                    bad = close[iou > nms_thres]

                    if len(bad) > 0:
                        mask = torch.ones(len(a)).type(torch.ByteTensor)
                        mask[bad] = 0
                        a = a[mask]
            pred = a

        # Experiment: Prior class size rejection
        # x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        # a = w * h  # area
        # ar = w / (h + 1e-16)  # aspect ratio
        # n = len(w)
        # log_w, log_h, log_a, log_ar = torch.log(w), torch.log(h), torch.log(a), torch.log(ar)
        # shape_likelihood = np.zeros((n, 60), dtype=np.float32)
        # x = np.concatenate((log_w.reshape(-1, 1), log_h.reshape(-1, 1)), 1)
        # from scipy.stats import multivariate_normal
        # for c in range(60):
        # shape_likelihood[:, c] =
        #   multivariate_normal.pdf(x, mean=mat['class_mu'][c, :2], cov=mat['class_cov'][c, :2, :2])

        class_prob, class_pred = torch.max(F.softmax(pred[:, 5:], 1), 1)

        # v = ((pred[:, 4] > conf_thres) & (class_prob > .4))  # TODO examine arbitrary 0.4 thres here
        v = pred[:, 4] > conf_thres
        v = v.nonzero().squeeze()
        if len(v.shape) == 0:
            v = v.unsqueeze(0)

        pred = pred[v]
        class_prob = class_prob[v]
        class_pred = class_pred[v]

        # If none are remaining => process next image
        nP = pred.shape[0]
        if not nP:
            continue

        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_prob, class_pred)
        detections = torch.cat((pred[:, :5], class_prob.float().unsqueeze(1), class_pred.float().unsqueeze(1)), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda(prediction.device)

        nms_style = 'OR'  # 'OR' (default), 'AND', 'MERGE' (experimental)
        for c in unique_labels:
            # Get the detections with class c
            dc = detections[detections[:, -1] == c]
            # Sort the detections by maximum object confidence
            _, conf_sort_index = torch.sort(dc[:, 4] * dc[:, 5], descending=True)
            dc = dc[conf_sort_index]

            # Non-maximum suppression
            det_max = []
            if nms_style == 'OR':  # default
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou(det_max[-1], dc[1:])  # iou with other boxes
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

                # Image      Total          P          R        mAP
                #  4964       5000      0.629      0.594      0.586

            elif nms_style == 'AND':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[:1], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'MERGE':  # weighted mixture box
                while len(dc) > 0:
                    iou = bbox_iou(dc[:1], dc[0:])  # iou with other boxes
                    i = iou > nms_thres

                    weights = dc[i, 4:5] * dc[i, 5:6]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[iou < nms_thres]

                # Image      Total          P          R        mAP
                #  4964       5000      0.633      0.598      0.589  # normal

            if len(det_max) > 0:
                det_max = torch.cat(det_max)
                # Add max detections to outputs
                output[image_i] = det_max if output[image_i] is None else torch.cat((output[image_i], det_max))

    return output

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # lists/pytorch to numpy
    tp, conf, pred_cls, target_cls = np.array(tp), np.array(conf), np.array(pred_cls), np.array(target_cls)

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = sum(target_cls == c)  # Number of ground truth objects
        n_p = sum(i)  # Number of predicted objects

        if (n_p == 0) and (n_gt == 0):
            continue
        elif (n_p == 0) or (n_gt == 0):
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = np.cumsum(1 - tp[i])
            tpc = np.cumsum(tp[i])

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(tpc[-1] / (n_gt + 1e-16))

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(tpc[-1] / (tpc[-1] + fpc[-1]))

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(ap), unique_classes.astype('int32'), np.array(r), np.array(p)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

class YOLOTest():
    def __init__(self, device, model):
        self.device = device
        self.model = model

    def train(self, train_dataloader, val_dataloader, epochs, optimizer, lr0, var=0, accumulated_batches=1):
        # Start training
        t0 = time.time()
        n_burnin = min(round(train_dataloader.nB / 5 + 1), 1000)  # number of burn-in batches
        start_epoch = 0
        best_loss = float('inf')
        best_map = -1.
        best_weights = None

        for epoch in range(epochs):
            epoch += start_epoch

            print(('%8s%12s' + '%10s' * 7) % (
                'Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))

            # Update scheduler (automatic)
            # scheduler.step()

            # Update scheduler (manual)  at 0, 54, 61 epochs to 1e-3, 1e-4, 1e-5
            if epoch > 50:
                lr = lr0 / 10
            else:
                lr = lr0
            for g in optimizer.param_groups:
                g['lr'] = lr

            ui = -1
            rloss = defaultdict(float)  # running loss
            optimizer.zero_grad()
            for i, (imgs, targets, _, _) in enumerate(train_dataloader):
                if sum([len(x) for x in targets]) < 1:  # if no targets continue
                    continue

                # SGD burn-in
                if (epoch == 0) & (i <= n_burnin):
                    lr = lr0 * (i / n_burnin) ** 4
                    for g in optimizer.param_groups:
                        g['lr'] = lr

                # Compute loss
                loss = self.model(imgs.to(self.device), targets, var=var)

                # Compute gradient
                loss.backward()

                # Accumulate gradient for x batches before optimizing
                if ((i + 1) % accumulated_batches == 0) or (i == len(train_dataloader) - 1):
                    optimizer.step()
                    optimizer.zero_grad()

                # Running epoch-means of tracked metrics
                ui += 1
                for key, val in self.model.losses.items():
                    rloss[key] = (rloss[key] * ui + val) / (ui + 1)

                s = ('%8s%12s' + '%10.3g' * 7) % (
                    '%g/%g' % (epoch, epochs - 1),
                    '%g/%g' % (i, len(train_dataloader) - 1),
                    rloss['xy'], rloss['wh'], rloss['conf'],
                    rloss['cls'], rloss['loss'],
                    self.model.losses['nT'], time.time() - t0)
                t0 = time.time()
                print(s)

            # Update best loss
            if rloss['loss'] < best_loss:
                best_loss = rloss['loss']

            # Save latest checkpoint
            checkpoint = {'epoch': epoch,
                        'best_loss': best_loss,
                        'model': self.model.state_dict(),
                        'optimizer': optimizer.state_dict()}

            weights = 'models' + os.sep
            latest = weights + 'latest.pt'
            best = weights + 'best.pt'
            torch.save(checkpoint, latest)

            # Save best checkpoint
            if best_loss == rloss['loss']:
                os.system('cp ' + latest + ' ' + best)

            # Save backup weights every 5 epochs (optional)
            # if (epoch > 0) & (epoch % 5 == 0):
            #     os.system('cp ' + latest + ' ' + weights + 'backup{}.pt'.format(epoch)))

            # Calculate mAP
            with torch.no_grad():
                mAP, R, P = self.test(val_dataloader)

                if mAP > best_map:
                    best_map = mAP
                    best_weights = copy.deepcopy(self.model.state_dict())

            # Write epoch results
            with open('results.txt', 'a') as file:
                file.write(s + '%11.3g' * 3 % (mAP, P, R) + '\n')
        
        return best_map, best_weights

    def test(self, dataloader, batch_size=16, img_size=416, iou_thres=0.5, conf_thres=0.3, nms_thres=0.45):
        nC = 80

        mean_mAP, mean_R, mean_P, seen = 0.0, 0.0, 0.0, 0
        print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
        outputs, mAPs, mR, mP, TP, confidence, pred_class, target_class, jdict = \
            [], [], [], [], [], [], [], [], []
        AP_accum, AP_accum_count = np.zeros(nC), np.zeros(nC)
        for batch_i, (imgs, targets, paths, shapes) in enumerate(dataloader):
            t = time.time()
            output = model(imgs.to(self.device))
            output = non_max_suppression(output, conf_thres=conf_thres, nms_thres=nms_thres)
            # Compute average precision for each sample
            for si, (labels, detections) in enumerate(zip(targets, output)):
                seen += 1

                if detections is None:
                    # If there are labels but no detections mark as zero AP
                    if labels.size(0) != 0:
                        mAPs.append(0), mR.append(0), mP.append(0)
                    continue

                # Get detections sorted by decreasing confidence scores
                detections = detections.cpu().numpy()
                detections = detections[np.argsort(-detections[:, 4])]

                # If no labels add number of detections as incorrect
                correct = []
                if labels.size(0) == 0:
                    # correct.extend([0 for _ in range(len(detections))])
                    mAPs.append(0), mR.append(0), mP.append(0)
                    continue
                else:
                    target_cls = labels[:, 0]

                    # Extract target boxes as (x1, y1, x2, y2)
                    target_boxes = xywh2xyxy(labels[:, 1:5]) * img_size

                    detected = []
                    for *pred_bbox, conf, obj_conf, obj_pred in detections:

                        pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
                        # Compute iou with target boxes
                        iou = bbox_iou(pred_bbox, target_boxes)
                        # Extract index of largest overlap
                        best_i = np.argmax(iou)
                        # If overlap exceeds threshold and classification is correct mark as correct
                        if iou[best_i] > iou_thres and obj_pred == labels[best_i, 0] and best_i not in detected:
                            correct.append(1)
                            detected.append(best_i)
                        else:
                            correct.append(0)

                # Compute Average Precision (AP) per class
                AP, AP_class, R, P = ap_per_class(tp=correct,
                                                conf=detections[:, 4],
                                                pred_cls=detections[:, 6],
                                                target_cls=target_cls)

                # Accumulate AP per class
                AP_accum_count += np.bincount(AP_class, minlength=nC)
                AP_accum += np.bincount(AP_class, minlength=nC, weights=AP)

                # Compute mean AP across all classes in this image, and append to image list
                mAPs.append(AP.mean())
                mR.append(R.mean())
                mP.append(P.mean())

                # Means of all images
                mean_mAP = np.mean(mAPs)
                mean_R = np.mean(mR)
                mean_P = np.mean(mP)

            # Print image mAP and running mean mAP
            print(('%11s%11s' + '%11.3g' * 4 + 's') %
                (seen, dataloader.nF, mean_P, mean_R, mean_mAP, time.time() - t))

        # Print mAP per class
        print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))

        # Return mAP
        return mean_mAP, mean_R, mean_P

if __name__ == '__main__':
    print("Loading model..")
    model = MaskedDarknet('./yolo.cfg')
    model.to('cuda:1')
    model.load_state_dict(torch.load('./models/yolov3.pt')["model"])
    print("Pruning model..")

    acc_threshold = .005
    prune_perc = 0.
    yolotester = YOLOTest('cuda:1', model)
    
    test_dataloader = LoadImagesAndLabels('/home/mlt/BayesThesis/masters-thesis-2019/yolodata/coco/5k.txt', batch_size=16, img_size=416)
    train_dataloader = LoadImagesAndLabels('/home/mlt/BayesThesis/masters-thesis-2019/yolodata/coco/train.txt', batch_size=16, img_size=416)
    val_dataloader = LoadImagesAndLabels('/home/mlt/BayesThesis/masters-thesis-2019/yolodata/coco/val.txt', batch_size=16, img_size=416)

    with torch.no_grad():
        curr_acc, _, _  = yolotester.test(val_dataloader)
        start_acc = copy.deepcopy(curr_acc)

    while (start_acc - curr_acc) < acc_threshold:
        prune_perc += 5.

        masks = weight_prune(model, prune_perc)
        for m in masks:
            for i in m:
                for j in i:
                    j.to('cuda:1')
        
        model.set_mask(masks)
        prune_rate(model)
        lr0 = 0.001

        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=.9)
        curr_acc, best_weights = yolotester.train(train_dataloader, val_dataloader, 3, optimizer, lr0)

        print("Loading best weights from training epochs..")
        model.load_state_dict(best_weights)
        print("Curr acc: ", curr_acc)
        print("Start acc: ", start_acc)
        print("Acc diff:", start_acc - curr_acc)

    with torch.no_grad():
        print("Testing model..")
        curr_acc, _, _ = yolotester.test(test_dataloader)
        print("Curr acc: ", curr_acc)
        print("Start acc: ", start_acc)
        print("Acc diff:", start_acc - curr_acc)

# conv_1 = MaskedConv2d(out_chanels=32, kernel_size=3, stride=1, padding=1)
# conv_1_activation = nn.LeakyReLU(0.1)

# Downsampling

# conv_2 = MaskedConv2d(out_channels=64, kernel_size=3, str)f
