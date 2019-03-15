import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import time
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
    def __init__(self, anchors, nC, img_size, yolo_layer, cfg):
        super(YOLOLayer, self).__init__()

        nA = len(anchors)
        self.anchors = torch.FloatTensor(anchors)
        self.nA = nA  # number of anchors (3)
        self.nC = nC  # number of classes (80)
        self.img_size = 0
        # self.coco_class_weights = coco_class_weights()


    def forward(self, p, img_size, targets=None, var=None):
        bs, nG = p.shape[0], p.shape[-1]

        if self.img_size != img_size:
            create_grids(self, img_size, nG)

            if p.is_cuda:
                self.grid_xy = self.grid_xy.cuda()
                self.anchor_wh = self.anchor_wh.cuda()

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
                txy, twh, mask, tcls = txy.cuda(), twh.cuda(), mask.cuda(), tcls.cuda()

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
                FT = torch.cuda.FloatTensor if p.is_cuda else torch.FloatTensor
                lxy, lwh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0])

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

class YOLOTest():
    def __init__(self, device, model):
        self.device = device
        self.model = model
        self.dataloader = LoadImagesAndLabels('./data/yolo/5k.txt', batch_size=16, img_size=416)
    
    def test(self, batch_size=16, img_size=416, iou_thres=0.5, conf_thres=0.3, nms_thres=0.45):
        nC = 80

        mean_mAP, mean_R, mean_P, seen = 0.0, 0.0, 0.0, 0
        print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
        outputs, mAPs, mR, mP, TP, confidence, pred_class, target_class, jdict = \
            [], [], [], [], [], [], [], [], []
        AP_accum, AP_accum_count = np.zeros(nC), np.zeros(nC)
        coco91class = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        for batch_i, (imgs, targets, paths, shapes) in enumerate(self.dataloader):
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

                if save_json:
                    # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                    box = torch.from_numpy(detections[:, :4]).clone()  # xyxy
                    scale_coords(img_size, box, shapes[si])  # to original shape
                    box = xyxy2xywh(box)  # xywh
                    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner

                    # add to json dictionary
                    for di, d in enumerate(detections):
                        jdict.append({
                            'image_id': int(Path(paths[si]).stem.split('_')[-1]),
                            'category_id': coco91class[int(d[6])],
                            'bbox': [float3(x) for x in box[di]],
                            'score': float3(d[4] * d[5])
                        })

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
                (seen, self.dataloader.nF, mean_P, mean_R, mean_mAP, time.time() - t))

        # Print mAP per class
        print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP') + '\n\nmAP Per Class:')

        for i, c in enumerate(load_classes(data_cfg_dict['names'])):
            print('%15s: %-.4f' % (c, AP_accum[i] / (AP_accum_count[i] + 1E-16)))

        # Return mAP
        return mean_mAP, mean_R, mean_P

if __name__ == '__main__':
    model = MaskedDarknet('./yolo.cfg')
    model.to('cuda:0')
    model.load_state_dict(torch.load('./models/yolov3.pt')['model'])
    masks = weight_prune(model, 80.)
    model.set_mask(masks)
    prune_rate(model)

    yolotester = YOLOTest('cuda:0', model)
    yolotester.test()


# conv_1 = MaskedConv2d(out_chanels=32, kernel_size=3, stride=1, padding=1)
# conv_1_activation = nn.LeakyReLU(0.1)

# Downsampling

# conv_2 = MaskedConv2d(out_channels=64, kernel_size=3, str)