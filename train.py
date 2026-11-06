"""Episodic UINO-FSS training and validation."""
import argparse
import itertools
import os

import monai
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf

from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from dataset.dataset import FSSDataset
from segment_anything_dinov2.build_fs_dino import _build_fs_dino


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def main_process(args):
    return args.rank == 0


def fix_bn(module):
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()


def poly_learning_rate(optimizer, base_lr, current_iter, max_iter, power=0.9):
    lr = base_lr * max(0.0, 1.0 - current_iter / max_iter) ** power
    for group in optimizer.param_groups:
        group['lr'] = lr


def train(args, epoch, model, dataloader, dataset, optimizer, training, train_sampler=None):
    meter = AverageMeter(dataset)
    model.train(training)
    utils.fix_randseed(args.train_seed + epoch if training else args.train_seed, not training)
    if training:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.apply(fix_bn)
    dice = monai.losses.DiceLoss(sigmoid=True, squared_pred=False, reduction='mean')
    bce = nn.BCEWithLogitsLoss(reduction='mean')
    limit = args.max_train_batches if training else args.max_val_batches
    steps = min(len(dataloader), limit) if limit else len(dataloader)
    if steps == 0:
        raise ValueError('The dataloader is empty; reduce batch size or check dataset metadata.')
    train_steps = min(args.train_batches, args.max_train_batches) if args.max_train_batches else args.train_batches
    totals = torch.zeros(2, device='cuda', dtype=torch.float64)
    for idx, batch in enumerate(itertools.islice(dataloader, steps)):
        batch = utils.to_cuda(batch)
        with torch.set_grad_enabled(training):
            logits, output = model(batch['query_img'], batch['support_imgs'], batch['support_masks'],
                                   args.stop_layer, args.is_light, args.is_sparse,
                                   args.ce_flag, args.is_ba, args.is_pcs)
            target = (batch['query_mask'] == 1).unsqueeze(1).float()
            loss = dice(output, target) + bce(output, target) + dice(logits, target) + bce(logits, target)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f'Non-finite loss at epoch {epoch}, batch {idx}: {loss.item()}')
            if training:
                current = epoch * train_steps + idx + 1
                poly_learning_rate(optimizer, args.learning_rate, current, args.epochs * train_steps, args.power)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
                optimizer.step()
                if main_process(args) and (args.max_train_batches or idx % 50 == 0):
                    Logger.info(f'Train step {idx + 1}/{steps}: loss={loss.item():.6f}, '
                                f'grad_norm={grad_norm.item():.6f}, lr={optimizer.param_groups[0]["lr"]:.8f}')
        prediction = torch.where(output.detach() > 0, 1, 0).squeeze(1)
        intersection, union = Evaluator.classify_prediction(prediction, batch)
        meter.update(intersection, union, batch['class_id'], loss.detach())
        totals[0] += loss.detach()
        totals[1] += 1
        if main_process(args):
            meter.write_process(idx, steps, epoch, write_batch_idx=1 if limit else 50)
    if args.distributed:
        for tensor in (meter.intersection_buf, meter.union_buf, totals):
            dist.all_reduce(tensor)
    avg_loss = totals[0] / totals[1]
    miou, fb_iou = meter.compute_iou()
    if main_process(args):
        Logger.info(f'{"Training" if training else "Validation"} epoch {epoch}: '
                    f'loss={avg_loss.item():.6f}, mIoU={miou.item():.2f}, FB-IoU={fb_iou.item():.2f}')
    return avg_loss, miou, fb_iou


def get_model(args):
    model = _build_fs_dino(OmegaConf.load('my_configs/vitb14_pretrain.yaml'))
    if args.load_distilled_model:
        checkpoint = torch.load(args.distilled_model_path, map_location='cpu')
        state = checkpoint.get('state_dict', checkpoint.get('net', checkpoint))
        model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=True)
    model.freeze_modules(args)
    model.cuda(args.local_rank)
    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,
                    find_unused_parameters=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    return model, optimizer


def parse_args():
    parser = argparse.ArgumentParser(description='UINO-FSS episodic training')
    parser.add_argument('--datapath', default='../')
    parser.add_argument('--benchmark', default='coco', choices=['pascal', 'coco'])
    parser.add_argument('--logpath', default='./log')
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--fold', type=int, default=0, choices=range(4))
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--bsz', type=int, default=4)
    parser.add_argument('--train_seed', type=int, default=321)
    parser.add_argument('--nshot', type=int, default=1, choices=[1])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--power', type=float, default=0.9)
    parser.add_argument('--index_split', type=int, default=0)
    parser.add_argument('--img_size', type=int, default=518, choices=[518])
    parser.add_argument('--stop_layer', type=int, default=4)
    parser.add_argument('--use_original_imgsize', action='store_true')
    for flag in ['is_light', 'is_sparse', 'ce_flag', 'is_ba', 'is_pcs',
                 'load_distilled_model', 'ba_train_flag', 'mask_decoder_train_flag', 'pe_train_flag']:
        parser.add_argument('--' + flag, action='store_true')
    parser.add_argument('--distilled_model_path', default='./checkpoints/distilled_base.pth')
    parser.add_argument('--max_train_batches', type=int, default=0, help='0 runs the full training epoch')
    parser.add_argument('--max_val_batches', type=int, default=0, help='0 runs full validation')
    args = parser.parse_args()
    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    args.rank = int(os.environ.get('RANK', '0'))
    args.distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    if args.epochs is None:
        args.epochs = 100 if args.benchmark == 'pascal' else 50
    if args.epochs <= 0 or args.bsz <= 0 or min(args.max_train_batches, args.max_val_batches) < 0:
        parser.error('Epochs and batch size must be positive; batch limits must be nonnegative.')
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for training.')
    torch.cuda.set_device(args.local_rank)
    if args.distributed:
        dist.init_process_group(backend='nccl')
    try:
        utils.fix_randseed(args.train_seed)
        if main_process(args):
            Logger.initialize(args, training=True)
        model, optimizer = get_model(args)
        if main_process(args):
            Logger.info(f'Total parameters: {sum(p.numel() for p in model.parameters())}; '
                        f'trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        Evaluator.initialize()
        FSSDataset.initialize(args.img_size, args.datapath, args.use_original_imgsize)
        val_loader, val_dataset, _ = FSSDataset.build_dataloader(
            args.benchmark, args.bsz, args.nworker, args.fold, 'val', args.nshot, 'lists', args.distributed)
        train_loader, train_dataset, sampler = FSSDataset.build_dataloader(
            args.benchmark, args.bsz, args.nworker, args.fold, 'train', args.nshot, 'lists', args.distributed)
        args.train_batches = len(train_loader)
        for epoch in range(args.epochs):
            trn_loss, trn_miou, trn_fb = train(args, epoch, model, train_loader, train_dataset,
                                             optimizer, True, sampler)
            val_loss, val_miou, val_fb = train(args, epoch, model, val_loader, val_dataset,
                                             optimizer, False)
            if main_process(args):
                Logger.save_model_miou(unwrap(model), epoch, val_miou, optimizer)
                for name, training_value, validation_value in [
                    ('loss', trn_loss, val_loss), ('miou', trn_miou, val_miou), ('fb_iou', trn_fb, val_fb)]:
                    Logger.tbd_writer.add_scalars('data/' + name,
                                                  {'trn': training_value.item(), 'val': validation_value.item()}, epoch)
                Logger.tbd_writer.flush()
        if main_process(args):
            Logger.tbd_writer.close()
            Logger.info('Finished Training')
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
