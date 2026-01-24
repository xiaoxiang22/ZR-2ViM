import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, base_dir='./data/', train=True, dataset='ISIC17', crop_szie=None):
        super(Data, self).__init__()
        self.dataset_dir = base_dir
        self.train = train
        self.dataset = dataset
        self.samples = []

        if self.dataset == 'ISIC17':
            if crop_szie is None:
                crop_szie = [256, 256]
            self.crop_size = crop_szie

            image_dir = os.path.join(self.dataset_dir, self.dataset, 'images')
            label_dir = os.path.join(self.dataset_dir, self.dataset, 'labels')
            if train:
                txt = os.path.join(self.dataset_dir, self.dataset, 'annotations', 'train.txt')
            else:
                txt = os.path.join(self.dataset_dir, self.dataset, 'annotations', 'test.txt')

            with open(txt, 'r') as f:
                filename_list = [x.strip() for x in f.readlines()]

            for filename in filename_list:
                image_path = os.path.join(image_dir, filename)
                label_path = os.path.join(label_dir, filename.replace('.jpg', '') + '_segmentation.png')

                if not os.path.exists(image_path):
                    print(f"Warning: Image file {image_path} does not exist.")
                    continue
                if not os.path.exists(label_path):
                    print(f"Warning: Label file {label_path} does not exist.")
                    continue

                self.samples.append((image_path, label_path, filename))

    def __len__(self):
        return len(self.samples)

    def _load_image_label(self, image_path, label_path):
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            img = np.array(Image.open(image_path).convert('RGB'))
        else:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        lbl = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if lbl is None:
            lbl = np.array(Image.open(label_path).convert('L'))
        return img, lbl

    def _resize_to_crop(self, img, lbl):
        target_w, target_h = self.crop_size[0], self.crop_size[1]
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        lbl = cv2.resize(lbl, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return img, lbl

    def __getitem__(self, index):
        image_path, label_path, name = self.samples[index]
        image, label = self._load_image_label(image_path, label_path)

        image, label = self._resize_to_crop(image, label)

        if self.train:
            if np.random.random() > 0.5:
                image = np.fliplr(image)
                label = np.fliplr(label)
            if np.random.random() > 0.5:
                image = np.flipud(image)
                label = np.flipud(label)

        if label.dtype != np.uint8:
            label = label.astype(np.uint8)
        label = (label > 127).astype(np.int64)

        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = image.transpose((2, 0, 1))

        if len(label.shape) > 2:
            label = label[:, :, 0]
        label = label.astype(np.int64)
        label = np.expand_dims(label, axis=0)

        sample = {'image': image, 'label': label, 'name': name}
        return sample
