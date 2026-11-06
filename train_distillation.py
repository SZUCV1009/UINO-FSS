import os
import itertools
import pdb
import numpy as np
import argparse
import random
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from segment_anything_dinov2.build_fs_dino import _build_fs_dino
from omegaconf import OmegaConf
from pathlib import Path
from segment_anything import sam_model_registry
from dataset.sa1b_dataset import transform, sa1b_dataset

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # dataset paths
    parser.add_argument('--dataset_path', type=str, default="../../sa_1b", help='root path of dataset')

    # training epochs, batch size and so on
    parser.add_argument('--epochs', type=int, default=15, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')

    # multi gpu settings
    parser.add_argument("--local-rank", "--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))

    # cuda settings
    # parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--deterministic', type=bool, default=True, help='deterministic')
    parser.add_argument('--benchmark', type=bool, default=False, help='benchmark')

    # learning process settings
    parser.add_argument('--optim', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # print and evaluate frequency during training
    parser.add_argument('--print_iters', type=int, default=200, help='print loss iterations')
    parser.add_argument('--eval_nums', type=int, default=48, help='evaluation numbers')
    parser.add_argument('--eval_iters', type=int, default=500, help='evaluation iterations')

    # file and folder paths
    parser.add_argument('--root_path', type=str, default="exp_loss", help='root path')
    parser.add_argument('--work_dir', type=str, default="work_dir", help='work directory')
    parser.add_argument('--save_dir', type=str, default="ckpt", help='save directory')
    parser.add_argument('--log_dir', type=str, default="log", help='save directory')
    parser.add_argument('--save_iters', type=int, default=5000, help='save iterations')

    parser.add_argument('--simg_size', type=int, default=592, help='sam img size')
    parser.add_argument('--dimg_size', type=int, default=518, help='dino img size')

    parser.add_argument('--ba_train_flag', action='store_true')
    parser.add_argument('--mask_decoder_train_flag', action='store_true')
    parser.add_argument('--pe_train_flag', action='store_true')

    parser.add_argument('--stop_layer', type=int, default=4, help='choose layer')
    parser.add_argument('--load_distilled_model', action='store_true')
    parser.add_argument('--distilled_model_path', type=str, default='')


    parser.add_argument('--max_train_batches', type=int, default=0)
    parser.add_argument('--max_val_batches', type=int, default=0)
    parser.add_argument('--max_train_samples', type=int, default=0)
    parser.add_argument('--check_updates', action='store_true')
    args = parser.parse_args()
    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    if not args.ba_train_flag:
        parser.error('Distillation requires --ba_train_flag.')
    if args.batch_size <= 0 or args.epochs <= 0:
        parser.error('Batch size and epochs must be positive.')
    if min(args.max_train_batches, args.max_val_batches, args.max_train_samples) < 0:
        parser.error('Batch/sample limits must be nonnegative.')
    if args.dimg_size % 14 or args.simg_size % 16 or args.dimg_size // 14 != args.simg_size // 16:
        parser.error('Use matching feature grids, e.g. SAM/DINO sizes 144/126, 288/252, or 592/518.')
    return args


def get_optimizer(args, model):
    if args.optim == 'adam':
        return optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        return optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        return optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(args.optim)


def get_scheduler(args, optimizer, total_steps):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)


def customized_mseloss(pred_feats, target_feats):
    return ((pred_feats - target_feats) ** 2).sum(1).mean().sqrt()


def test(args, model, test_loader, image_encoder):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for idx, [dino_imgs,sam_imgs] in enumerate(itertools.islice(test_loader, args.max_val_batches or len(test_loader))):
            dino_imgs = dino_imgs.cuda(args.local_rank)
            sam_imgs = sam_imgs.cuda(args.local_rank)
            dino_imgs = dino_imgs.cuda(args.local_rank)
            sam_imgs = sam_imgs.cuda(args.local_rank)
            with torch.no_grad():
                target_feats = image_encoder(sam_imgs)
            pred_feats = model(dino_imgs, stop_layer=args.stop_layer, distill_only=True)
            test_loss += customized_mseloss(pred_feats, target_feats).item()

    return test_loss / min(len(test_loader), args.max_val_batches or len(test_loader))


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def main(args):
    # multi gpu settings
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    if 'RANK' not in os.environ:
        raise RuntimeError('Launch with: torchrun --standalone --nproc_per_node=1 train_distillation.py ...')
    torch.distributed.init_process_group(backend='nccl')
    if args.batch_size % dist.get_world_size():
        raise ValueError('Global batch_size must be divisible by the process count.')

    # file folder creating
    if args.local_rank == 0:
        if not os.path.exists(os.path.join(args.root_path, args.work_dir, args.save_dir)):
            os.makedirs(os.path.join(args.root_path, args.work_dir, args.save_dir))

        if not os.path.exists(os.path.join(args.root_path, args.work_dir, args.log_dir)):
            os.makedirs(os.path.join(args.root_path, args.work_dir, args.log_dir))

    # seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = args.deterministic
        cudnn.benchmark = args.benchmark

    # dataset
    train_dirs = ["sa_" + str(i).zfill(6) for i in range(10)]
    val_dirs = ['sa_000010']
    train_dataset = sa1b_dataset(args.dataset_path, train_dirs, transform, args.max_train_samples or None, args.simg_size, args.dimg_size)
    val_dataset = sa1b_dataset(args.dataset_path, val_dirs, transform, args.eval_nums, args.simg_size, args.dimg_size)
    # training sampler
    train_sampler = DistributedSampler(train_dataset)
    # data loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size // dist.get_world_size(),
                              shuffle=(train_sampler is None), num_workers=args.num_workers, sampler=train_sampler,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    if args.local_rank == 0:
        writer = SummaryWriter(os.path.join(args.root_path, args.work_dir, args.log_dir))

    # model
    cfg = OmegaConf.load("my_configs/vitb14_pretrain.yaml")
    model = _build_fs_dino(cfg)
    model.to(device)
    if args.load_distilled_model:
        path = args.distilled_model_path
        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get('state_dict', checkpoint.get('net', checkpoint))
        model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=True)
    model.requires_grad_(False)
    model.ba.requires_grad_(True)
    initial_parameters = {name: p.detach().cpu().clone() for name, p in model.named_parameters()} if args.check_updates and args.local_rank == 0 else None
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank, find_unused_parameters=False)

    # ImageEncoder
    path = os.path.join("checkpoints/sam_vit_h_4b8939.pth")
    sam = sam_model_registry["vit_h"](checkpoint=path)
    image_encoder = sam.image_encoder
    image_encoder = image_encoder.cuda(args.local_rank).eval().requires_grad_(False)

    # optimizer and scheduler
    optimizer = get_optimizer(args, model)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_scheduler(args, optimizer, total_steps)

    total_iters = 0

    for epoch in range(1, args.epochs + 1):
        # new epoch
        if args.local_rank == 0:
            print("------start epoch {}------".format(epoch))
        train_sampler.set_epoch(epoch)

        if args.local_rank == 0:
            pbar = tqdm(total=min(len(train_loader), args.max_train_batches or len(train_loader)), leave=False, desc='train')
        else:
            pbar = None

        for batch_idx, [dino_imgs,sam_imgs] in enumerate(itertools.islice(train_loader, args.max_train_batches or len(train_loader))):
            model.train()
            model.module.image_encoder.eval()
            total_iters += 1

            dino_imgs = dino_imgs.cuda(args.local_rank)
            sam_imgs = sam_imgs.cuda(args.local_rank)
            with torch.no_grad():
                target_feats = image_encoder(sam_imgs)

            optimizer.zero_grad()
            pred_feats = model(dino_imgs, stop_layer=args.stop_layer, distill_only=True)
            loss = customized_mseloss(pred_feats, target_feats)

            if not torch.isfinite(loss).item():
                raise FloatingPointError(f'Non-finite distillation loss at step {total_iters}')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], float('inf'), error_if_nonfinite=True)
            optimizer.step()
            if args.local_rank == 0 and args.max_train_batches:
                print(f'Distillation step {total_iters}: loss={loss.item():.6f}, grad_norm={grad_norm.item():.6f}', flush=True)
            loss = reduce_mean(loss, dist.get_world_size())

            # if is master process
            if args.local_rank == 0:
                # print training info
                if (batch_idx + 1) % args.print_iters == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE Loss: {:.6f}'.format(
                        epoch, batch_idx * len(sam_imgs) * dist.get_world_size(), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item()))
                    writer.add_scalar("mse_loss", loss.item(), total_iters)

            # evaluation
            if total_iters % args.eval_iters == 0:
                test_loss = test(args, model, val_loader, image_encoder)
                if args.local_rank == 0:
                    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
                    writer.add_scalar("eval_mse_loss", test_loss, total_iters)

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        dist.barrier()
        scheduler.step()

        if args.local_rank == 0:
            torch.save(
                {'epoch': epoch, 'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict()},
                os.path.join(args.root_path, args.work_dir, args.save_dir, "epoch_" + str(epoch) + ".pth"))

    # save final model
    if args.local_rank == 0:
        torch.save(model.module.state_dict(),
                   os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_final.pth"))
        writer.close()
        if initial_parameters is not None:
            changed = [name for name, p in model.module.named_parameters()
                       if not torch.equal(p.detach().cpu(), initial_parameters[name])]
            assert changed and all(name.startswith('ba.') for name in changed), changed
            print(f'Parameter update check: {len(changed)} BA tensors changed; frozen parameters unchanged.', flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_option()
    main(args)