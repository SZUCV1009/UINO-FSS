""" Dataloader builder for few-shot semantic segmentation dataset  """
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset.pascal import FewShotPASCAL
from dataset.coco import FewShotCOCO
import albumentations as A
import albumentations.pytorch
import numpy as np
import cv2
from dataset import transform
from torch.utils.data.distributed import DistributedSampler


class Compose(A.Compose):
    def __init__(self, transforms, bbox_params=None, keypoint_params=None, additional_targets=None, p=1):
        super().__init__(transforms, bbox_params=bbox_params, keypoint_params=keypoint_params,
                         additional_targets=additional_targets, p=p)
        self.is_check_shapes = False

    def __call__(self, image, mask):
        augmented = super().__call__(image=np.array(image), mask=np.array(mask))
        return augmented['image'], augmented['mask']

class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):

        cls.datasets = {
            'pascal': FewShotPASCAL,
            'coco': FewShotCOCO,
        }
        cats_augmentation = [
            A.ToGray(p=0.2),
            A.Posterize(p=0.2),
            A.Equalize(p=0.2),
            A.Sharpen(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.Solarize(p=0.2),
            A.ColorJitter(p=0.2),
        ]

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]

        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        cls.splits_trn_transform = Compose([
            *cats_augmentation,
            A.Resize(img_size, img_size),
            A.Normalize(cls.img_mean, cls.img_std),
            A.pytorch.transforms.ToTensorV2(),
        ])

        cls.splits_transform = Compose([
            A.Resize(img_size, img_size),
            A.Normalize(cls.img_mean, cls.img_std),
            A.pytorch.transforms.ToTensorV2(),
        ])

        cls.img_mean = [0.485*255, 0.456*255, 0.406*255]
        cls.img_std = [0.229*255, 0.224*255, 0.225*255]

        cls.lists_val_transform = transform.Compose([
            transform.Resize(img_size),
            transform.ToTensor(),
            transform.Normalize(mean=cls.img_mean, std=cls.img_std)])

        scale_min = 0.8
        scale_max = 1.25
        rotate_min = -10
        rotate_max = 10
        padding_label = 255

        cls.lists_train_transform = transform.Compose([
            transform.RandScale([scale_min, scale_max]),
            transform.RandRotate([rotate_min, rotate_max], padding=cls.img_mean, ignore_label=padding_label),
            transform.RandomGaussianBlur(),
            transform.RandomHorizontalFlip(),
            transform.Resize([img_size, img_size]),
            transform.ToTensor(),
            transform.Normalize(mean=cls.img_mean, std=cls.img_std)])



    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, mode, shot=1, split_type='lists', distributed=False):
        nworker = nworker if mode in ['train', 'trn'] else 0

        if split_type == 'lists':
                _transform = cls.lists_train_transform if mode == 'train' else cls.lists_val_transform
        else:
            _transform = cls.splits_trn_transform if mode == 'trn' else cls.splits_transform


        # =========================
        # 3️⃣ 构建 dataset
        # =========================
        dataset = cls.datasets[benchmark](
            datapath=cls.datapath,
            split_type=split_type,
            mode=mode,
            fold=fold,
            shot=shot,
            transform=_transform,
            use_original_imgsize=cls.use_original_imgsize
        )

        train_sampler = DistributedSampler(dataset) if distributed else None

        # =========================
        # 4️⃣ dataloader
        # =========================
        dataloader = DataLoader(
            dataset,
            batch_size=bsz,
            num_workers=nworker,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True if mode in ['train', 'trn'] else False,
            shuffle=False
        )

        return dataloader, dataset, train_sampler
