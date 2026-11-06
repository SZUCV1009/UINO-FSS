import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import cv2
from torch.nn import functional as F
from torch.utils.data.distributed import DistributedSampler

from segment_anything.utils.transforms import ResizeLongestSide


class sa1b_dataset(Dataset):
    def __init__(self, root_path, img_dirs, transformer, max_num=None, simg_size=592, dimg_size=518):
        self.root_path = root_path
        self.img_dirs = img_dirs
        self.transformer = transformer
        self.max_num = max_num
        self.img_paths = []
        for i, img_dir in enumerate(img_dirs):
            img_names = sorted(os.listdir(os.path.join(root_path, img_dir)))
            self.img_paths += [os.path.join(root_path, img_dir, img_name) for img_name in img_names if
                               ".jpg" in img_name]
            if max_num and len(self.img_paths) >= max_num:
                self.img_paths = self.img_paths[:max_num]
                break
        self.simg_size = simg_size
        self.dimg_size = dimg_size

    def __len__(self):
        if not self.max_num:
            return len(self.img_paths)
        return min(self.max_num, len(self.img_paths))

    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transformer:
            dino_img = self.transformer(img, img_size=self.dimg_size)
            sam_img = self.transformer(img, img_size=self.simg_size)

        # feat = np.load(self.img_paths[index].replace(".jpg", ".npy")).squeeze()

        # return img, feat, self.img_paths[index].replace(".jpg", ".json")
        return [dino_img, sam_img]


def transform(x, img_size=896):
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    transform = ResizeLongestSide(img_size)
    x = transform.apply_image(x)
    x = torch.as_tensor(x)
    x = x.permute(2, 0, 1).contiguous()

    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def get_sa1b_dataloaders(transformer, root_path, train_dirs, val_dirs, batch_size=4, num_workers=4, val_max_num=1000):
    train_set = sa1b_dataset(root_path, train_dirs, transformer)
    val_set = sa1b_dataset(root_path, val_dirs, transformer, val_max_num)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, )

    return train_loader, val_loader
