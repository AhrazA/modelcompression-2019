import torch
import torch.nn as nn
import torch.nn.functional as F
from .masked_sequential import MaskedSequential
from .masked_conv_2d import MaskedConv2d
from collections import defaultdict

class MaskedModuleList(nn.ModuleList):
    def __init__(self):
        super(MaskedModuleList, self).__init__()
    
    def set_mask(self, masks):
        for module, mask in zip(self.children(), masks[0]):
            module.set_mask(mask)


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
    def __init__(self, anchors, nC, img_size, yolo_layer, cfg, device='cuda:0'):
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

    def __init__(self, cfg_path, img_size=416, device='cuda:0'):
        super(MaskedDarknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg_path)
        self.module_defs[0]['cfg'] = cfg_path
        self.module_defs[0]['height'] = img_size
        self.hyperparams, self.module_list = create_modules(self.module_defs, device)
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

def create_modules(module_defs, device):
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
            yolo_layer = YOLOLayer(anchors, nC, img_size, yolo_layer_count, cfg=hyperparams['cfg'], device=device)
            modules.add_module('yolo_%d' % i, yolo_layer)
            yolo_layer_count += 1

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list

def create_grids(self, img_size, nG):
    self.stride = img_size / nG

    # build xy offsets
    grid_x = torch.arange(nG).repeat((nG, 1)).view((1, 1, nG, nG)).float()
    grid_y = grid_x.permute(0, 1, 3, 2)
    self.grid_xy = torch.stack((grid_x, grid_y), 4)

    # build wh gains
    self.anchor_vec = self.anchors / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.nA, 1, 1, 2)

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

def parse_model_cfg(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs