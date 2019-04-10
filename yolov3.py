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
from classifier_utils import setup_default_args
from pruning.masked_conv_2d import MaskedConv2d
from pruning.masked_linear import MaskedLinear
from pruning.masked_sequential import MaskedSequential
from pruning.methods import weight_prune, prune_rate
from pruning.masked_yolo_v3 import MaskedDarknet, MaskedModuleList

from yolo_imageloader import LoadImagesAndLabels

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

class YoloWrapper():
    def __init__(self, device, model):
        self.device = device
        self.model = model
        model.to(device)

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
            output = self.model(imgs.to(self.device))
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

        # Return mAP
        return mean_mAP, mean_R, mean_P