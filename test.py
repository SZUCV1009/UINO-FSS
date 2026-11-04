import os
import argparse

import torch.nn.functional as F
import torch.nn as nn
import torch

from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common import utils

from dataset.dataset import FSSDataset
from omegaconf import OmegaConf
import numpy as np

@torch.no_grad()
def test(model, dataloader, nshot, test_seed=0):
    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(test_seed)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        batch = utils.to_cuda(batch)
        input = batch['query_img']
        s_input = batch['support_imgs']
        s_mask = batch['support_masks']

        model.eval()
        logits, output = model(input, s_input, s_mask, args.stop_layer, args.is_light, args.is_sparse, args.ce_flag, args.is_ba, args.is_pcs)

        pred_mask = torch.where(output>0,1,0)
        pred_mask = pred_mask.squeeze(1)

        assert pred_mask.size() == batch['query_mask'].size()

        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask, batch['class_id'], idx,
                                                  area_inter[1].float() / area_union[1].float())


    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='UINO-FSS Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../')
    parser.add_argument('--benchmark', type=str, default='coco', choices=['pascal', 'coco'])
    parser.add_argument('--logpath', type=str, default='./log_test')
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--nshot', type=int, default=1, choices=[1], help='This release supports 1-shot evaluation.')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--use_original_imgsize', action='store_true')
    parser.add_argument('--img_size', type=int, default=518, choices=[518])
    parser.add_argument('--checkpoint', type=str, default=None, help='Defaults to final_<benchmark>/fold<fold>.pth')

    parser.add_argument('--stop_layer', type=int, default=4)
    parser.add_argument('--is_light', action='store_true')
    parser.add_argument('--is_sparse', action='store_true')
    parser.add_argument('--ce_flag', action='store_true')
    parser.add_argument('--is_ba', action='store_true')
    parser.add_argument('--is_pcs', action='store_true')

    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('CUDA is required for evaluation. Install CUDA-enabled PyTorch and check your GPU driver.')
    path = args.checkpoint or os.path.join(f'final_{args.benchmark}', f'fold{args.fold}.pth')
    if not os.path.isfile(path):
        parser.error(f'Evaluation checkpoint not found: {path}. See README.md for checkpoint preparation.')
    Logger.initialize(args, training=False)
    Logger.info(f'Evaluation checkpoint: {path}')
    from segment_anything_dinov2.build_fs_dino import _build_fs_dino

    # Model initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load("my_configs/vitb14_pretrain.yaml")
    model = _build_fs_dino(cfg)
    checkpoint = torch.load(path, map_location="cpu")
    if 'net' in checkpoint:
        state_dict = checkpoint['net']
    else:
        state_dict = checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # Logger.log_params(model)
    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize)

    # Dataset initialization
    FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    dataloader_test, dataset_val, _ = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val', args.nshot, 'splits')

    seed_num = 1
    seed_array = [0]

    mIoU_array = np.zeros(seed_num)
    FBIoU_array = np.zeros(seed_num)
    for i in range(seed_num):
        seed_val = seed_array[i]
        test_miou, test_fb_iou = test(model, dataloader_test, args.nshot, test_seed=seed_val)
        mIoU_array[i] = test_miou
        FBIoU_array[i] = test_fb_iou

        Logger.info('Fold %d mIoU: %.2f \t FB-IoU: %.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))


    print('Seed:  {}'.format(seed_array))
    print('mIoU:  {}'.format(np.round(mIoU_array, 2)))
    print('FBIoU: {}'.format(np.round(FBIoU_array, 2)))

    print('Best_mIoU: {:.2f} \t Best_FBIoU: {:.2f} '.format(mIoU_array.max(), FBIoU_array.max()))
    print('Mean_mIoU: {:.2f} \t Mean_FBIoU: {:.2f} '.format(mIoU_array.mean(), FBIoU_array.mean()))
