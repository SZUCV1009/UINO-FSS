import os
import cv2
import random
import pickle
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class FewShotCOCO(Dataset):

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
        self.benchmark = 'coco'

        self.nclass = 80
        self.nfolds = 4

        self.base_path = os.path.join(datapath, 'COCO2014')

        # self.img_path = os.path.join(datapath, 'VOC2012/JPEGImages/')
        # self.ann_path = os.path.join(datapath, 'VOC2012/SegmentationClassAug/')

        self.class_ids = self._build_class_ids()
        self.class_names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'trafficlight',
              'firehydrant', 'stopsign', 'parkingmeter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
              'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sportsball', 'kite', 'baseballbat', 'baseballglove', 'skateboard', 'surfboard',
              'tennisracket', 'bottle', 'wineglass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hotdog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
              'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
              'cellphone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddybear', 'hairdrier', 'toothbrush']

        if self.split_type == 'lists':
            self.class_list = list(range(1, 81))
            if self.fold == 3:
                self.sub_val_list = list(range(4, 81, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
            elif self.fold == 2:
                self.sub_val_list = list(range(3, 80, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
            elif self.fold == 1:
                self.sub_val_list = list(range(2, 79, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
            elif self.fold == 0:
                self.sub_val_list = list(range(1, 78, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))

            self.data_list, self.sub_class_file_list = self._load_lists_metadata()
        else:
            self.img_metadata_classwise = self.build_img_metadata_classwise()
            self.img_metadata = self.build_img_metadata()

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
        class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        class_ids = class_ids_trn if self.mode in ['train', 'trn'] else class_ids_val

        return class_ids

    # ==========================
    # LISTS
    # ==========================
    def _resolve_metadata_path(self, path):
        marker = 'COCO2014/'
        relative = path.split(marker, 1)[1] if marker in path else path.lstrip('./')
        return os.path.join(self.base_path, relative)

    def _load_lists_metadata(self):
        mode_folder = 'train' if self.mode == 'train' else 'val'
        list_root = os.path.join('./lists/coco/fss_list', mode_folder)

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
        image_path, label_path = self.data_list[idx]
        query_name = os.path.basename(image_path)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_image = image.copy()
        image = np.float32(image)
        img_cv2 = image.copy()
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
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

        # for cls in range(1,self.num_classes+1):
        #     select_pix = np.where(label_b_tmp == cls)
        #     if cls in self.sub_list:
        #         label_b[select_pix[0],select_pix[1]] = self.sub_list.index(cls) + 1
        #     else:
        #         label_b[select_pix[0],select_pix[1]] = 0

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
            support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
            target_pix = np.where(support_label == class_chosen)
            ignore_pix = np.where(support_label == 255)
            support_label[:, :] = 0
            support_label[target_pix[0], target_pix[1]] = 1
            support_label_mask = support_label.copy()
            # support_label, support_label_mask = transform_anns(support_label, self.ann_type)  # mask/bbox
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
            image, label = self.transform(image, label)  # transform the triple
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
    def _getitem_splits(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # (due to the large size of the COCO dataset)
        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame()
        ori_image = np.array(query_img)
        ori_image = cv2.resize(ori_image, (518, 518), interpolation=cv2.INTER_NEAREST)
        ori_support_img = np.array(support_imgs[0])
        ori_support_img = cv2.resize(ori_support_img, (518, 518), interpolation=cv2.INTER_NEAREST)

        query_img, query_mask = self.transform(query_img, query_mask)
        query_mask = query_mask.float()
        if not self.use_original_imgsize:
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_transformed = [self.transform(support_img, support_cmask) for support_img, support_cmask in zip(support_imgs, support_masks)]
        support_masks = [x[1] for x in support_transformed]
        support_imgs = torch.stack([x[0] for x in support_transformed])

        for midx, smask in enumerate(support_masks):
            support_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
        support_masks = torch.stack(support_masks)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,
                 'ori_image': ori_image,
                 'ori_support_img': ori_support_img,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'class_id': torch.tensor(class_sample),
                 'class_name': self.class_names[torch.tensor(class_sample)]}

        return batch


    def build_img_metadata_classwise(self):
        with open('./splits/coco/%s/fold%d.pkl' % (self.mode, self.fold), 'rb') as f:
            img_metadata_classwise = pickle.load(f)
        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += self.img_metadata_classwise[k]
        return sorted(list(set(img_metadata)))

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, 'annotations', name)
        mask = torch.tensor(np.array(Image.open(mask_path[:mask_path.index('.jpg')] + '.png')))
        return mask

    def load_frame(self):
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
        query_mask = self.read_mask(query_name)

        org_qry_imsize = query_img.size

        query_mask[query_mask != class_sample + 1] = 0
        query_mask[query_mask == class_sample + 1] = 1

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        support_imgs = []
        support_masks = []
        for support_name in support_names:
            support_imgs.append(Image.open(os.path.join(self.base_path, support_name)).convert('RGB'))
            support_mask = self.read_mask(support_name)
            support_mask[support_mask != class_sample + 1] = 0
            support_mask[support_mask == class_sample + 1] = 1
            support_masks.append(support_mask)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize
