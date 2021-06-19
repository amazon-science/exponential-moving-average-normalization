# some code in this file is adapted from
# https://github.com/pytorch/examples
# Original Copyright 2017. Licensed under the BSD 3-Clause License.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import builtins
import math
import os
import shutil
import time
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

import data.datasets as datasets
import backbone as backbone_models
from models import get_fixmatch_model
from utils import utils, lr_schedule, get_norm, dist_utils
import data.transforms as data_transforms
from engine import validate

backbone_model_names = sorted(name for name in backbone_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(backbone_models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--trainindex_x', default=None, type=str, metavar='PATH',
                    help='path to train annotation_x (default: None)')
parser.add_argument('--trainindex_u', default=None, type=str, metavar='PATH',
                    help='path to train annotation_u (default: None)')
parser.add_argument('--arch', metavar='ARCH', default='FixMatch',
                    help='model architecture')
parser.add_argument('--backbone', default='resnet50_encoder',
                    choices=backbone_model_names,
                    help='model architecture: ' +
                        ' | '.join(backbone_model_names) +
                        ' (default: resnet50_encoder)')
parser.add_argument('--cls', default=1000, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--warmup-epoch', default=0, type=int, metavar='N',
                    help='number of epochs for learning warmup')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--nesterov', action='store_true', default=False,
                    help='use nesterov momentum')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', default=1, type=int,
                    metavar='N', help='evaluation epoch frequency (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained model (default: none)')
parser.add_argument('--self-pretrained', default='', type=str, metavar='PATH',
                    help='path to MoCo pretrained model (default: none)')
parser.add_argument('--super-pretrained', default='', type=str, metavar='PATH',
                    help='path to supervised pretrained model (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# FixMatch configs:
parser.add_argument('--anno-percent', type=float, default=0.1,
                    help='number of labeled data')
parser.add_argument('--split-seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--mu', default=5, type=int,
                    help='coefficient of unlabeled batch size')
parser.add_argument('--lambda-u', default=10, type=float,
                    help='coefficient of unlabeled loss')
parser.add_argument('--threshold', default=0.7, type=float,
                    help='pseudo label threshold')
parser.add_argument('--eman', action='store_true', default=False,
                    help='use EMAN')
parser.add_argument('--ema-m', default=0.999, type=float,
                    help='EMA decay rate')
parser.add_argument('--weak-type', default='DefaultTrain', type=str,
                    help='the type for strong augmentation')
parser.add_argument('--strong-type', default='RandAugment', type=str,
                    help='the type for strong augmentation')
parser.add_argument('--norm', default='None', type=str,
                    help='the normalization for backbone (default: None)')
# online_net.backbone for BYOL
parser.add_argument('--model-prefix', default='encoder_q', type=str,
                    help='the model prefix of self-supervised pretrained state_dict')

best_acc1 = 0


def main():
    args = parser.parse_args()
    assert args.warmup_epoch < args.schedule[0]
    print(args)

    if args.seed is not None:
        seed = args.seed + dist_utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        # random.seed(seed)

    assert 0 < args.anno_percent < 1

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    print("=> creating model '{}' with backbone '{}'".format(args.arch, args.backbone))
    model_func = get_fixmatch_model(args.arch)
    norm = get_norm(args.norm)
    model = model_func(
        backbone_models.__dict__[args.backbone],
        eman=args.eman,
        momentum=args.ema_m,
        norm=norm
    )
    print(model)

    if args.self_pretrained:
        if os.path.isfile(args.self_pretrained):
            print("=> loading checkpoint '{}'".format(args.self_pretrained))
            checkpoint = torch.load(args.self_pretrained, map_location="cpu")

            # rename self pre-trained keys to model.main keys
            state_dict = checkpoint['state_dict']
            model_prefix = 'module.' + args.model_prefix
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith(model_prefix) and not k.startswith(model_prefix + '.fc'):
                    # replace prefix
                    new_key = k.replace(model_prefix, "main.backbone")
                    state_dict[new_key] = state_dict[k]
                    if model.ema is not None:
                        new_key = k.replace(model_prefix, "ema.backbone")
                        state_dict[new_key] = state_dict[k].clone()
                # delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            if len(msg.missing_keys) > 0:
                print("missing keys:\n{}".format('\n'.join(msg.missing_keys)))
            if len(msg.unexpected_keys) > 0:
                print("unexpected keys:\n{}".format('\n'.join(msg.unexpected_keys)))
            print("=> loaded pre-trained model '{}' (epoch {})".format(args.self_pretrained, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.self_pretrained))
    elif args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading pretrained model from '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                new_key = k.replace("module.", "")
                state_dict[new_key] = state_dict[k]
                del state_dict[k]
            model_num_cls = state_dict['fc.weight'].shape[0]
            if model_num_cls != args.cls:
                # if num_cls don't match, remove the last layer
                del state_dict['fc.weight']
                del state_dict['fc.bias']
                msg = model.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}, \
                    "missing keys:\n{}".format('\n'.join(msg.missing_keys))
            else:
                model.load_state_dict(state_dict)
            print("=> loaded pre-trained model '{}' (epoch {})".format(args.pretrained, checkpoint['epoch']))
        else:
            print("=> no pretrained model found at '{}'".format(args.pretrained))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Supervised Data loading code
    if args.trainindex_x is not None and args.trainindex_u is not None:
        print("load index from {}/{}".format(args.trainindex_x, args.trainindex_u))
        index_info_x = os.path.join(args.data, 'indexes', args.trainindex_x)
        index_info_u = os.path.join(args.data, 'indexes', args.trainindex_u)
        index_info_x = pd.read_csv(index_info_x)
        trainindex_x = index_info_x['Index'].tolist()
        index_info_u = pd.read_csv(index_info_u)
        trainindex_u = index_info_u['Index'].tolist()
        train_dataset_x, train_dataset_u, val_dataset = get_imagenet_ssl(
            args.data, trainindex_x, trainindex_u,
            weak_type=args.weak_type, strong_type=args.strong_type)
    else:
        print("random sampling {} percent of data".format(args.anno_percent * 100))
        train_dataset_x, train_dataset_u, val_dataset = get_imagenet_ssl_random(
            args.data, args.anno_percent, weak_type=args.weak_type, strong_type=args.strong_type)
    print("train_dataset_x:\n{}".format(train_dataset_x))
    print("train_dataset_u:\n{}".format(train_dataset_u))
    print("val_dataset:\n{}".format(val_dataset))

    # Data loading code
    train_sampler = DistributedSampler if args.distributed else RandomSampler

    train_loader_x = DataLoader(
        train_dataset_x,
        sampler=train_sampler(train_dataset_x),
        batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    train_loader_u = DataLoader(
        train_dataset_u,
        sampler=train_sampler(train_dataset_u),
        batch_size=args.batch_size * args.mu,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    best_epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.epochs):
        if epoch >= args.warmup_epoch:
            lr_schedule.adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader_x, train_loader_u, model, optimizer, epoch, args)

        is_best = False
        if (epoch + 1) % args.eval_freq == 0:
            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, args)
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:
                best_epoch = epoch

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best)

    print('Best Acc@1 {0} @ epoch {1}'.format(best_acc1, best_epoch + 1))


def train(train_loader_x, train_loader_u, model, optimizer, epoch, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    losses_x = utils.AverageMeter('Loss_x', ':.4e')
    losses_u = utils.AverageMeter('Loss_u', ':.4e')
    top1_x = utils.AverageMeter('Acc_x@1', ':6.2f')
    top5_x = utils.AverageMeter('Acc_x@5', ':6.2f')
    top1_u = utils.AverageMeter('Acc_u@1', ':6.2f')
    top5_u = utils.AverageMeter('Acc_u@5', ':6.2f')
    mask_info = utils.AverageMeter('Mask', ':6.3f')
    curr_lr = utils.InstantMeter('LR', '')
    progress = utils.ProgressMeter(
        len(train_loader_u),
        [curr_lr, batch_time, data_time, losses, losses_x, losses_u, top1_x, top5_x, top1_u, top5_u, mask_info],
        prefix="Epoch: [{}/{}]\t".format(epoch, args.epochs))

    epoch_x = epoch * math.ceil(len(train_loader_u) / len(train_loader_x))
    if args.distributed:
        print("set epoch={} for labeled sampler".format(epoch_x))
        train_loader_x.sampler.set_epoch(epoch_x)
        print("set epoch={} for unlabeled sampler".format(epoch))
        train_loader_u.sampler.set_epoch(epoch)

    train_iter_x = iter(train_loader_x)
    # switch to train mode
    model.train()
    if args.eman:
        print("setting the ema model to eval mode")
        if hasattr(model, 'module'):
            model.module.ema.eval()
        else:
            model.ema.eval()

    end = time.time()
    for i, (images_u, targets_u) in enumerate(train_loader_u):
        try:
            images_x, targets_x = next(train_iter_x)
        except Exception:
            epoch_x += 1
            print("reshuffle train_loader_x at epoch={}".format(epoch_x))
            if args.distributed:
                print("set epoch={} for labeled sampler".format(epoch_x))
                train_loader_x.sampler.set_epoch(epoch_x)
            train_iter_x = iter(train_loader_x)
            images_x, targets_x = next(train_iter_x)

        images_u_w, images_u_s = images_u
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images_x = images_x.cuda(args.gpu, non_blocking=True)
            images_u_w = images_u_w.cuda(args.gpu, non_blocking=True)
            images_u_s = images_u_s.cuda(args.gpu, non_blocking=True)
        targets_x = targets_x.cuda(args.gpu, non_blocking=True)
        targets_u = targets_u.cuda(args.gpu, non_blocking=True)

        # warmup learning rate
        if epoch < args.warmup_epoch:
            warmup_step = args.warmup_epoch * len(train_loader_u)
            curr_step = epoch * len(train_loader_u) + i + 1
            lr_schedule.warmup_learning_rate(optimizer, curr_step, warmup_step, args)
        curr_lr.update(optimizer.param_groups[0]['lr'])

        # model forward
        logits_x, logits_u_w, logits_u_s = model(images_x, images_u_w, images_u_s)
        # pseudo label
        pseudo_label = torch.softmax(logits_u_w.detach_(), dim=-1)
        max_probs, pseudo_targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(args.threshold).float()

        # compute losses
        loss_x = F.cross_entropy(logits_x, targets_x, reduction='mean')
        loss_u = (F.cross_entropy(logits_u_s, pseudo_targets_u, reduction='none') * mask).mean()
        loss = loss_x + args.lambda_u * loss_u

        # measure accuracy and record loss
        losses.update(loss.item())
        losses_x.update(loss_x.item(), images_x.size(0))
        losses_u.update(loss_u.item(), images_u_w.size(0))
        acc1_x, acc5_x = utils.accuracy(logits_x, targets_x, topk=(1, 5))
        top1_x.update(acc1_x[0], logits_x.size(0))
        top5_x.update(acc5_x[0], logits_x.size(0))
        acc1_u, acc5_u = utils.accuracy(logits_u_w, targets_u, topk=(1, 5))
        top1_u.update(acc1_u[0], logits_u_w.size(0))
        top5_u.update(acc5_u[0], logits_u_w.size(0))
        mask_info.update(mask.mean().item(), mask.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update the ema model
        if args.eman:
            if hasattr(model, 'module'):
                model.module.momentum_update_ema()
            else:
                model.momentum_update_ema()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def get_imagenet_ssl(image_root, trainindex_x, trainindex_u,
                     train_type='DefaultTrain', val_type='DefaultVal', weak_type='DefaultTrain',
                     strong_type='RandAugment'):
    traindir = os.path.join(image_root, 'train')
    valdir = os.path.join(image_root, 'val')
    transform_x = data_transforms.get_transforms(train_type)
    weak_transform = data_transforms.get_transforms(weak_type)
    strong_transform = data_transforms.get_transforms(strong_type)
    transform_u = data_transforms.TwoCropsTransform(weak_transform, strong_transform)
    transform_val = data_transforms.get_transforms(val_type)

    train_dataset_x = datasets.ImageFolderWithIndex(
        traindir, trainindex_x, transform=transform_x)

    train_dataset_u = datasets.ImageFolderWithIndex(
        traindir, trainindex_u, transform=transform_u)

    val_dataset = datasets.ImageFolder(
        valdir, transform=transform_val)

    return train_dataset_x, train_dataset_u, val_dataset


def x_u_split(labels, percent, num_classes):
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        label_per_class = max(1, round(percent * len(idx)))
        np.random.shuffle(idx)
        labeled_idx.extend(idx[:label_per_class])
        unlabeled_idx.extend(idx[label_per_class:])
    print('labeled_idx ({}): {}, ..., {}'.format(len(labeled_idx), labeled_idx[:5], labeled_idx[-5:]))
    print('unlabeled_idx ({}): {}, ..., {}'.format(len(unlabeled_idx), unlabeled_idx[:5], unlabeled_idx[-5:]))
    return labeled_idx, unlabeled_idx


def get_imagenet_ssl_random(image_root, percent, train_type='DefaultTrain',
                            val_type='DefaultVal', weak_type='DefaultTrain', strong_type='RandAugment'):
    traindir = os.path.join(image_root, 'train')
    valdir = os.path.join(image_root, 'val')
    transform_x = data_transforms.get_transforms(train_type)
    weak_transform = data_transforms.get_transforms(weak_type)
    strong_transform = data_transforms.get_transforms(strong_type)
    transform_u = data_transforms.TwoCropsTransform(weak_transform, strong_transform)
    transform_val = data_transforms.get_transforms(val_type)

    base_dataset = datasets.ImageFolder(traindir)

    train_idxs_x, train_idxs_u = x_u_split(
        base_dataset.targets, percent, len(base_dataset.classes))

    train_dataset_x = datasets.ImageFolderWithIndex(
        traindir, train_idxs_x, transform=transform_x)

    train_dataset_u = datasets.ImageFolderWithIndex(
        traindir, train_idxs_u, transform=transform_u)

    val_dataset = datasets.ImageFolder(
        valdir, transform=transform_val)

    return train_dataset_x, train_dataset_u, val_dataset


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    main()
