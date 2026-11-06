import os
import cv2
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class FewShotPASCAL(Dataset):

    def __init__(self, datapath, split_type='lists', mode='train', fold=0, shot=1,
                 transform=None, use_original_imgsize=False):

        assert split_type in ['lists', 'splits']

        if split_type == 'lists':
            allowed_modes = ['train', 'val']
        else:
            allowed_modes = ['trn', 'val']

        assert mode in allowed_modes

        self.datapath = datapath
        self.split_type = split_type
        self.mode = mode
        self.fold = fold
        self.shot = shot
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize
        self.benchmark = 'pascal'

        self.nclass = 20
        self.nfolds = 4

        self.img_path = os.path.join(datapath, 'VOCdevkit/VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'VOCdevkit/VOC2012/SegmentationClassAug/')

        self.class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                            'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                            'train', 'tvmonitor']

        self.class_ids = self._build_class_ids()

        if self.split_type == 'lists':
            self.class_list = list(range(1, 21))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            if self.fold == 3:
                self.sub_list = list(range(1, 16))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                self.sub_val_list = list(range(16, 21))  # [16,17,18,19,20]
            elif self.fold == 2:
                self.sub_list = list(range(1, 11)) + list(range(16, 21))  # [1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
                self.sub_val_list = list(range(11, 16))  # [11,12,13,14,15]
            elif self.fold == 1:
                self.sub_list = list(range(1, 6)) + list(range(11, 21))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(6, 11))  # [6,7,8,9,10]
            elif self.fold == 0:
                self.sub_list = list(range(6, 21))  # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(1, 6))  # [1,2,3,4,5]

            self.data_list, self.sub_class_file_list = self._load_lists_metadata()
        else:
            self.img_metadata = self._load_splits_metadata()
            self.img_metadata_classwise = self._build_classwise_metadata()

    # --------------------------
    # Public
    # --------------------------
    def __len__(self):
        if self.split_type == 'splits':
            return len(self.img_metadata) if self.mode == 'trn' else 1000
        return len(self.data_list) if self.mode == 'train' else 1000


    def __getitem__(self, idx):
        if self.split_type == 'lists':
            return self._getitem_lists(idx)
        else:
            return self._getitem_splits(idx)

    # --------------------------
    # Class Split
    # --------------------------
    def _build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]

        if self.mode in ['train', 'trn']:
            return class_ids_trn
        else:
            return class_ids_val

    # ==========================
    # LISTS
    # ==========================
    def _resolve_metadata_path(self, path):
        marker = 'VOC2012/'
        relative = path.split(marker, 1)[1] if marker in path else path.lstrip('./')
        return os.path.join(os.path.join(self.datapath, 'VOCdevkit', 'VOC2012'), relative)

    def _load_lists_metadata(self):
        mode_folder = 'train' if self.mode == 'train' else 'val'
        list_root = os.path.join('./lists/pascal/fss_list', mode_folder)

        data_list_file = os.path.join(list_root, f'data_list_{self.fold}.txt')
        sub_class_file = os.path.join(list_root, f'sub_class_file_list_{self.fold}.txt')

        with open(data_list_file, 'r') as f:
            lines = f.readlines()
        metadata = [(l.split(' ')[0], l.split(' ')[1].strip()) for l in lines]

        with open(sub_class_file, 'r') as f:
            class_file_list = eval(f.read())

        metadata = [(self._resolve_metadata_path(image), self._resolve_metadata_path(mask))
                    for image, mask in metadata]
        class_file_list = {
            key: [(self._resolve_metadata_path(image), self._resolve_metadata_path(mask))
                  for image, mask in entries]
            for key, entries in class_file_list.items()
        }
        return metadata, class_file_list

    def _getitem_lists(self, idx):
        idx %= len(self.data_list)
        image_path, label_path = self.data_list[idx]
        query_name = os.path.basename(image_path)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_image = image.copy()
        image = np.float32(image)
        img_cv2 = image.copy()
        '''label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)'''
        label = np.array(Image.open(label_path))
        ori_label = label.copy()
        # label_b = cv2.imread(os.path.join(self.base_path, label_path.split('/')[-1]), cv2.IMREAD_GRAYSCALE)


        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)
        new_label_class = []

        for c in label_class:
            if c in self.sub_val_list:
                if self.mode == 'val':
                    new_label_class.append(c)
            if c in self.sub_list:
                if self.mode == 'train':
                    new_label_class.append(c)
        label_class = new_label_class
        assert len(label_class) > 0

        class_chosen = label_class[random.randint(1, len(label_class)) - 1]

        target_pix = np.where(label == class_chosen)
        ignore_pix = np.where(label == 255)
        label[:, :] = 0
        if target_pix[0].shape[0] > 0:
            label[target_pix[0], target_pix[1]] = 1
        label[ignore_pix[0], ignore_pix[1]] = 255

        file_class_chosen = self.sub_class_file_list[class_chosen]
        num_file = len(file_class_chosen)

        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []
        for k in range(self.shot):
            support_idx = random.randint(1, num_file) - 1
            support_image_path = image_path
            support_label_path = label_path
            while ((
                support_image_path == image_path and support_label_path == label_path) or support_idx in support_idx_list):
                support_idx = random.randint(1, num_file) - 1
                support_image_path, support_label_path = file_class_chosen[support_idx]
            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        support_image_list_ori = []
        support_label_list_ori = []
        support_label_list_ori_mask = []
        subcls_list = []
        if self.mode == 'train':
            subcls_list.append(self.sub_list.index(class_chosen))
        else:
            subcls_list.append(self.sub_val_list.index(class_chosen))
        for k in range(self.shot):
            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k]
            support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)
            support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
            support_image = np.float32(support_image)
            support_label = np.array(Image.open(support_label_path))
            target_pix = np.where(support_label == class_chosen)
            ignore_pix = np.where(support_label == 255)
            support_label[:, :] = 0
            support_label[target_pix[0], target_pix[1]] = 1
            # support_label, support_label_mask = transform_anns(support_label, self.ann_type)  # mask/bbox
            support_label_mask = support_label
            support_label[ignore_pix[0], ignore_pix[1]] = 255
            support_label_mask[ignore_pix[0], ignore_pix[1]] = 255
            if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
                raise (RuntimeError(
                    "Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))
            support_image_list_ori.append(support_image)
            support_label_list_ori.append(support_label)
            support_label_list_ori_mask.append(support_label_mask)
        assert len(support_label_list_ori) == self.shot and len(support_image_list_ori) == self.shot

        raw_image = image.copy()
        raw_label = label.copy()
        # raw_label_b = label_b.copy()
        support_image_list = [[] for _ in range(self.shot)]
        support_label_list = [[] for _ in range(self.shot)]
        if self.transform is not None:
            # image, label, label_b, img_cv2 = self.transform_tri(image, label, label_b, img_cv2)  # transform the triple
            image, label = self.transform(image, label)
            for k in range(self.shot):
                support_image_list[k], support_label_list[k] = self.transform(support_image_list_ori[k],
                                                                              support_label_list_ori[k])

        s_xs = support_image_list
        s_ys = support_label_list
        s_x = s_xs[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
        s_y = s_ys[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)

        ori_image = cv2.resize(ori_image, (518,518), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
        ori_label = cv2.resize(ori_label, (518, 518), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)

        query_ignore_idx = torch.where(label==255,1,0)
        gt_mask = torch.where(label==1,1,0)
        s_y = torch.where(s_y==1,1,0)

        batch = {'query_img': image,
                 'query_mask': gt_mask,
                 'query_name': query_name,
                 'query_ignore_idx': query_ignore_idx,
                 'ori_img': ori_image,
                 'support_imgs': s_x,
                 'support_masks': s_y,
                 'class_id': torch.tensor(class_chosen-1),
                 'class_name': self.class_names[torch.tensor(class_chosen-1)]}

        # Return
        # if self.mode == 'train':
        #     return image, query_name, label, s_x, s_y, subcls_list, class_chosen-1, ori_image
        # elif self.mode == 'val':
        #     return image, query_name, label, s_x, s_y, subcls_list, class_chosen-1, raw_label, raw_label_b, ori_image
        return batch

    # ==========================
    # SPLITS
    # ==========================
    def _load_splits_metadata(self):
        def read_metadata(split, fold_id):
            fold_n_metadata = os.path.join('./splits/pascal/%s/fold%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.mode in ['train', 'trn']:  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.mode, fold_id)
        elif self.mode == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.mode, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.mode)

        print('Total (%s) images are : %d' % (self.mode, len(img_metadata)))

        return img_metadata


    def _build_classwise_metadata(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise

    def _getitem_splits(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name, support_names)
        ori_img = np.array(query_img)
        ori_img = cv2.resize(ori_img, (518, 518), interpolation=cv2.INTER_NEAREST)
        ori_support_img = np.array(support_imgs)
        ori_support_img = cv2.resize(ori_support_img[0], (518, 518), interpolation=cv2.INTER_NEAREST)

        query_img, query_cmask = self.transform(query_img, query_cmask)
        if not self.use_original_imgsize:
            query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask.float(), class_sample)

        support_transformed = [self.transform(support_img, support_cmask) for support_img, support_cmask in zip(support_imgs, support_cmasks)]
        support_cmasks = [x[1] for x in support_transformed]
        support_imgs = torch.stack([x[0] for x in support_transformed])

        support_masks = []
        support_ignore_idxs = []
        for scmask in support_cmasks:
            scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_mask, support_ignore_idx = self.extract_ignore_idx(scmask, class_sample)
            support_masks.append(support_mask)
            support_ignore_idxs.append(support_ignore_idx)
        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,
                 'query_ignore_idx': query_ignore_idx,
                 'ori_img': ori_img,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'support_ignore_idxs': support_ignore_idxs,
                 'ori_support_img':ori_support_img,

                 'class_id': torch.tensor(class_sample),
                 'class_name': self.class_names[torch.tensor(class_sample)]}

        return batch

    def sample_episode(self, idx):
        query_name, class_sample = self.img_metadata[idx]

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_sample


    def extract_ignore_idx(self, mask, class_id):
        boundary = (mask / 255).floor()
        mask[mask != class_id + 1] = 0
        mask[mask == class_id + 1] = 1

        return mask, boundary


    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs = [self.read_img(name) for name in support_names]
        support_masks = [self.read_mask(name) for name in support_names]

        org_qry_imsize = query_img.size

        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png')))
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')
