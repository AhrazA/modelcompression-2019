from fasterrcnn.utils.config import cfg
from torch.utils.data.sampler import Sampler
import torch
from fasterrcnn.roi_data_layer.roidb import combined_roidb
from fasterrcnn.roi_data_layer.roibatchLoader import roibatchLoader
from torch.autograd import Variable
from fasterrcnn.utils.net_utils import adjust_learning_rate
import time


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
        self.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        model.to(device)
    
    def train(self, batch_size, lr, epochs):
        imdb, roidb, ratio_list, ratio_index = combined_roidb(self.imdb_name)
        train_size = len(roidb)

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
        disp_interval = 100

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
                im_data.data.resize_(data[0].size()).copy_(data[0])
                im_info.data.resize_(data[1].size()).copy_(data[1])
                gt_boxes.data.resize_(data[2].size()).copy_(data[2])
                num_boxes.data.resize_(data[3].size()).copy_(data[3])

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
