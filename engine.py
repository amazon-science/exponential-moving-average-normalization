# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import time
from datetime import timedelta
import faiss
import numpy as np

import torch
import torch.nn as nn

from utils import utils


def validate(val_loader, model, criterion, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    progress = utils.ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.avg:.4f}'
              .format(top1=top1, top5=top5, loss=losses))

    return top1.avg


def ss_validate(val_loader_base, val_loader_query, model, args):
    print("start KNN evaluation with key size={} and query size={}".format(
        len(val_loader_base.dataset.targets), len(val_loader_query.dataset.targets)))
    batch_time_key = utils.AverageMeter('Time', ':6.3f')
    batch_time_query = utils.AverageMeter('Time', ':6.3f')
    # switch to evaluate mode
    model.eval()

    feats_base = []
    target_base = []
    feats_query = []
    target_query = []

    with torch.no_grad():
        start = time.time()
        end = time.time()
        # Memory features
        for i, (images, target) in enumerate(val_loader_base):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute features
            feats = model(images)
            # L2 normalization
            feats = nn.functional.normalize(feats, dim=1)

            feats_base.append(feats)
            target_base.append(target)

            # measure elapsed time
            batch_time_key.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Extracting key features: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    i, len(val_loader_base), batch_time=batch_time_key))

        end = time.time()
        for i, (images, target) in enumerate(val_loader_query):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute features
            feats = model(images)
            # L2 normalization
            feats = nn.functional.normalize(feats, dim=1)

            feats_query.append(feats)
            target_query.append(target)

            # measure elapsed time
            batch_time_query.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Extracting query features: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    i, len(val_loader_query), batch_time=batch_time_query))

        feats_base = torch.cat(feats_base, dim=0)
        target_base = torch.cat(target_base, dim=0)
        feats_query = torch.cat(feats_query, dim=0)
        target_query = torch.cat(target_query, dim=0)
        feats_base = feats_base.detach().cpu().numpy()
        target_base = target_base.detach().cpu().numpy()
        feats_query = feats_query.detach().cpu().numpy()
        target_query = target_query.detach().cpu().numpy()
        feat_time = time.time() - start

        # KNN search
        index = faiss.IndexFlatL2(feats_base.shape[1])
        index.add(feats_base)
        D, I = index.search(feats_query, args.num_nn)
        preds = np.array([np.bincount(target_base[n]).argmax() for n in I])

        NN_acc = (preds == target_query).sum() / len(target_query) * 100.0
        knn_time = time.time() - start - feat_time
        print("finished KNN evaluation, feature time: {}, knn time: {}".format(
            timedelta(seconds=feat_time), timedelta(seconds=knn_time)))
        print(' * NN Acc@1 {:.3f}'.format(NN_acc))

    return NN_acc
