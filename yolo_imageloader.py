import torch
import numpy as np
import math
import cv2
import os

def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y

def letterbox(img, height=416, color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh

class LoadImagesAndLabels:  # for training
    def __init__(self, path, batch_size=1, img_size=608, multi_scale=False, augment=False):
        with open(path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [x.replace('\n', '') for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [x.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')
                            for x in self.img_files]

        self.nF = len(self.img_files)  # number of image files
        self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.batch_size = batch_size
        self.height = img_size
        self.multi_scale = multi_scale
        self.augment = augment

        assert self.nF > 0, 'No images found in %s' % path

    def __iter__(self):
        self.count = -1
        self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nB:
            raise StopIteration

        ia = self.count * self.batch_size
        ib = min((self.count + 1) * self.batch_size, self.nF)

        if self.multi_scale:
            # Multi-Scale YOLO Training
            height = random.choice(range(10, 20)) * 32  # 320 - 608 pixels
        else:
            # Fixed-Scale YOLO Training
            height = self.height

        img_all, labels_all, img_paths, img_shapes = [], [], [], []
        for index, files_index in enumerate(range(ia, ib)):
            img_path = self.img_files[self.shuffled_vector[files_index]]
            label_path = self.label_files[self.shuffled_vector[files_index]]

            img = cv2.imread(img_path)  # BGR
            assert img is not None, 'File Not Found ' + img_path

            augment_hsv = True
            if self.augment and augment_hsv:
                # SV augmentation by 50%
                fraction = 0.50
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                S = img_hsv[:, :, 1].astype(np.float32)
                V = img_hsv[:, :, 2].astype(np.float32)

                a = (random.random() * 2 - 1) * fraction + 1
                S *= a
                if a > 1:
                    np.clip(S, a_min=0, a_max=255, out=S)

                a = (random.random() * 2 - 1) * fraction + 1
                V *= a
                if a > 1:
                    np.clip(V, a_min=0, a_max=255, out=V)

                img_hsv[:, :, 1] = S.astype(np.uint8)
                img_hsv[:, :, 2] = V.astype(np.uint8)
                cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

            h, w, _ = img.shape
            img, ratio, padw, padh = letterbox(img, height=height)

            # Load labels
            if os.path.isfile(label_path):
                labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 5)

                # Normalized xywh to pixel xyxy format
                labels = labels0.copy()
                labels[:, 1] = ratio * w * (labels0[:, 1] - labels0[:, 3] / 2) + padw
                labels[:, 2] = ratio * h * (labels0[:, 2] - labels0[:, 4] / 2) + padh
                labels[:, 3] = ratio * w * (labels0[:, 1] + labels0[:, 3] / 2) + padw
                labels[:, 4] = ratio * h * (labels0[:, 2] + labels0[:, 4] / 2) + padh
            else:
                labels = np.array([])

            # Augment image and labels
            if self.augment:
                img, labels, M = random_affine(img, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))

            plotFlag = False
            if plotFlag:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 10)) if index == 0 else None
                plt.subplot(4, 4, index + 1).imshow(img[:, :, ::-1])
                plt.plot(labels[:, [1, 3, 3, 1, 1]].T, labels[:, [2, 2, 4, 4, 2]].T, '.-')
                plt.axis('off')

            nL = len(labels)
            if nL > 0:
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5].copy()) / height

            if self.augment:
                # random left-right flip
                lr_flip = True
                if lr_flip & (random.random() > 0.5):
                    img = np.fliplr(img)
                    if nL > 0:
                        labels[:, 1] = 1 - labels[:, 1]

                # random up-down flip
                ud_flip = False
                if ud_flip & (random.random() > 0.5):
                    img = np.flipud(img)
                    if nL > 0:
                        labels[:, 2] = 1 - labels[:, 2]

            img_all.append(img)
            labels_all.append(torch.from_numpy(labels))
            img_paths.append(img_path)
            img_shapes.append((h, w))

        # Normalize
        img_all = np.stack(img_all)[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB and cv2 to pytorch
        img_all = np.ascontiguousarray(img_all, dtype=np.float32)
        img_all /= 255.0

        return torch.from_numpy(img_all), labels_all, img_paths, img_shapes

    def __len__(self):
        return self.nB  # number of batches

