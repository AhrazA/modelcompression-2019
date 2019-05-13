from fasterrcnn.utils.config import cfg, get_output_dir
from torch.utils.data.sampler import Sampler
import torch
import os
from fasterrcnn.roi_data_layer.roidb import combined_roidb
from fasterrcnn.roi_data_layer.roibatchLoader import roibatchLoader
from torch.autograd import Variable
from fasterrcnn.utils.net_utils import adjust_learning_rate
from fasterrcnn.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from fasterrcnn.roi_layers import nms
import time
import numpy as np
import pickle
from easydict import EasyDict as edict
import pprint

def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  for k, v in a.items():
    # a must specify keys that are in b
    if k not in b:
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                          'for config key: {}').format(type(b[k]),
                                                       type(v), k))

    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print(('Error under config key: {}'.format(k)))
        raise
    else:
      b[k] = v

def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f))

  _merge_a_into_b(yaml_cfg, cfg)

def cfg_from_list(cfg_list):
  """Set config keys via list (e.g., from command line)."""
  from ast import literal_eval
  assert len(cfg_list) % 2 == 0
  for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
    key_list = k.split('.')
    d = cfg
    for subkey in key_list[:-1]:
      assert subkey in d
      d = d[subkey]
    subkey = key_list[-1]
    assert subkey in d
    try:
      value = literal_eval(v)
    except:
      # handle the case when v is a string literal
      value = v
    assert type(value) == type(d[subkey]), \
      'type {} does not match original type {}'.format(
        type(value), type(d[subkey]))
    d[subkey] = value

class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

class FasterRCNNWrapper():
    def __init__(self, device, model):
        self.device = device
        self.model = model
        self.imdb_name = "voc_2007_trainval"
        self.imdbval_name = "voc_2007_test"
        self.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
        model.to(device)
    
    def train(self, batch_size, lr, epochs):
        imdb, roidb, ratio_list, ratio_index = combined_roidb(self.imdb_name)
        train_size = len(roidb)

        self.set_cfgs += ['MAX_NUM_GT_BOXES', '20']
        cfg_from_list(self.set_cfgs)
        cfg_from_file('/mnt/home/a318599/Bayesnn/masters-thesis-2019/fasterrcnn/res101.yml')

        print('{:d} roidb entries'.format(len(roidb)))
        sampler_batch = sampler(train_size, batch_size)
        dataset = roibatchLoader(roidb, ratio_list, ratio_index, batch_size, \
                           imdb.num_classes, training=True)
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                            sampler=sampler_batch, num_workers=1)

        im_data = torch.FloatTensor(1).to(self.device)
        im_info = torch.FloatTensor(1).to(self.device)
        num_boxes = torch.LongTensor(1).to(self.device)
        gt_boxes = torch.FloatTensor(1).to(self.device)

        im_data = Variable(im_data)
        im_info = Variable(im_info)
        num_boxes = Variable(num_boxes)
        gt_boxes = Variable(gt_boxes)

        fasterRCNN = self.model

        params = []

        for key, value in dict(fasterRCNN.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
        iters_per_epoch = int(train_size / batch_size)
        lr_decay_step = 5
        lr_decay_gamma = 0.1
        disp_interval = 10

        best_map = -1.
        best_weights = None

        for epoch in range(epochs):
            # setting to train mode
            fasterRCNN.train()
            loss_temp = 0
            start = time.time()

            if epoch % (lr_decay_step + 1) == 0:
                adjust_learning_rate(optimizer, lr_decay_gamma)
                lr *= lr_decay_gamma

            data_iter = iter(dataloader)

            for step in range(iters_per_epoch):
                data = next(data_iter)
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes.resize_(data[2].size()).copy_(data[2])
                num_boxes.resize_(data[3].size()).copy_(data[3])

                fasterRCNN.zero_grad()
                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

                loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                    + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
                loss_temp += loss.item()

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % disp_interval == 0:
                    end = time.time()
                    if step > 0:
                        loss_temp /= (disp_interval + 1)

                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                    print("[epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                            % (epoch, step, iters_per_epoch, loss_temp, lr))
                    print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
                    print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                                % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                    
                    loss_temp = 0
                    start = time.time()

            curr_mAP = self.test()
            if curr_mAP > best_map:
                best_weights = self.model.state_dict()
                best_map = curr_mAP
        
        return best_map, best_weights

    def test(self):
        cfg_from_file('/mnt/home/a318599/Bayesnn/masters-thesis-2019/fasterrcnn/res101.yml')
        cfg_from_list(self.set_cfgs)        
        cfg.TRAIN.USE_FLIPPED = False
        
        imdb, roidb, ratio_list, ratio_index = combined_roidb(self.imdbval_name, False)
        imdb.competition_mode(on=True)

        print('{:d} roidb entries'.format(len(roidb)))

        pprint.pprint(cfg)

        np.random.seed(cfg.RNG_SEED)

        im_data = torch.FloatTensor(1).to(self.device)
        im_info = torch.FloatTensor(1).to(self.device)
        num_boxes = torch.LongTensor(1).to(self.device)
        gt_boxes = torch.FloatTensor(1).to(self.device)

        im_data = Variable(im_data)
        im_info = Variable(im_info)
        num_boxes = Variable(num_boxes)
        gt_boxes = Variable(gt_boxes)

        save_name = 'faster_rcnn_10'
        num_images = len(imdb.image_index)
        all_boxes = [[[] for _ in range(num_images)]
                    for _ in range(imdb.num_classes)]

        output_dir = get_output_dir(imdb, save_name)
        dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                                imdb.num_classes, training=False, normalize = False)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                    shuffle=False, num_workers=1)

        data_iter = iter(dataloader)

        _t = {'im_detect': time.time(), 'misc': time.time()}
        det_file = os.path.join(output_dir, 'detections.pkl')

        fasterRCNN = self.model

        fasterRCNN.eval()

        thresh = 0.0
        max_per_image = 100
        start = time.time()

        empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
        for i in range(num_images):

            data = next(data_iter)
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])

            det_tic = time.time()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                    # if args.class_agnostic:
                    #     box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                    #             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    #     box_deltas = box_deltas.view(1, -1, 4)
                    # else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= data[1][0][2].item()

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            det_toc = time.time()
            detect_time = det_toc - det_tic
            misc_tic = time.time()

            for j in range(1, imdb.num_classes):
                inds = torch.nonzero(scores[:,j]>thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:,j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    # if args.class_agnostic:
                    #     cls_boxes = pred_boxes[inds, :]
                    # else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                    
                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    # if vis:
                    #     im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                    all_boxes[j][i] = cls_dets.cpu().numpy()
                else:
                    all_boxes[j][i] = empty_array

            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1]
                                            for j in range(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, imdb.num_classes):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]

            misc_toc = time.time()
            nms_time = misc_toc - misc_tic

            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                .format(i + 1, num_images, detect_time, nms_time))

        # if vis:
        #     cv2.imwrite('result.png', im2show)
        #     pdb.set_trace()
        #     #cv2.imshow('test', im2show)
        #     #cv2.waitKey(0)

        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections')
        aps = imdb.evaluate_detections(all_boxes, output_dir)

        end = time.time()
        print("test time: %0.4fs" % (end - start))

        return sum(aps) / len(aps)