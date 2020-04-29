import argparse
import os
import time
import shutil
import pickle
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm

from dataset import TSNDataSet
from models import TSN
from transforms import *
from opts import parser
import pandas as pd
import numpy as np
from tqdm import tqdm

best_prec1 = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    global args, best_prec1
    args = parser.parse_args()
    
    if not os.path.exists(args.record_path + args.modality.lower()):
        os.mkdir(args.record_path + args.modality.lower())

    num_class = 2

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type, dropout=args.dropout, partial_bn=not args.no_partialbn)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()

    # model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    model = model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_set = TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl="frame_{:06d}.jpg" if args.modality in ["RGB", "RGBDiff"] else "{:06d}.jpg",
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ]))
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_set = TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl="frame_{:06d}.jpg" if args.modality in ["RGB", "RGBDiff"] else "{:06d}.jpg",
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ]))
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer

    criterion = torch.nn.CrossEntropyLoss().cuda()


    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1, pred_dict = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            with open(args.record_path + args.modality.lower() + '/' + args.modality.lower() + '_video_preds.pickle','wb') as f:
                pickle.dump(pred_dict,f)
                f.close()
                
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    if args.no_partialbn:
        model.partialBN(False)
    else:
        model.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    progress = tqdm(train_loader)
    for i, (keys, input, target) in enumerate(progress):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input = input.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # if args.clip_gradient is not None:
        #     total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            # if total_norm > args.clip_gradient:
            #     print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    info = {'Epoch': [epoch],
            'Batch Time': [round(batch_time.avg, 3)],
            'Epoch Time': [round(batch_time.sum, 3)],
            'Data Time': [round(data_time.avg, 3)],
            'Loss': [round(losses.avg.item(), 5)],
            'Prec@1': [round(top1.avg.item(), 4)],
            'lr': optimizer.param_groups[0]['lr']
            }
    record_info(info, args.record_path + args.modality.lower() + '/' + args.modality.lower() + '_train.csv', 'train')

def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    data_time = AverageMeter()

    # switch to evaluate mode
    model.eval()
    dic_video_level_preds={}

    end = time.time()
    progress = tqdm(val_loader)
    with torch.no_grad():
        for i, (keys, input, target) in enumerate(progress):
            target = target.cuda()
            input = input.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))
            losses.update(loss.data, input.size(0))
            top1.update(prec1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            pred = output.data.cpu().numpy()
            for j in range(pred.shape[0]):
                videoName = keys[j]
                if videoName not in dic_video_level_preds.keys():
                    dic_video_level_preds[videoName] = pred[j,:]
                else:
                    print('amazing')
                    dic_video_level_preds[videoName] += pred[j,:]
                

    info = {'Epoch': [epoch],
            'Batch Time': [round(batch_time.avg, 3)],
            'Epoch Time': [round(batch_time.sum, 3)],
            'Data Time': [round(data_time.avg, 3)],
            'Loss': [round(losses.avg.item(), 5)],
            'Prec@1': [round(top1.avg.item(), 4)],
            }
    record_info(info, args.record_path + args.modality.lower() + '/' + args.modality.lower() + '_test.csv', 'test')
    # print(('Testing Results: Prec@1 {top1.avg:.3f} Loss {loss.avg:.5f}'
    #       .format(top1=top1, loss=losses)))

    return top1.avg, dic_video_level_preds


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = args.record_path + args.modality.lower() + '/' + '_'.join((args.snapshot_pref, args.modality.lower(), filename))
    torch.save(state, filename)
    if is_best:
        best_name = args.record_path + args.modality.lower() + '/' + '_'.join((args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def record_info(info, filename, mode):
    if mode == 'train':
        result = (
            'Batch Time {batch_time} '
            'Epoch Time {epoch_time} '
            'Data {data_time} \n'
            'Loss {loss} '
            'Prec@1 {top1} '
            'LR {lr}\n'.format(batch_time=info['Batch Time'], epoch_time=info['Epoch Time'],
                               data_time=info['Data Time'], loss=info['Loss'],
                               top1=info['Prec@1'], lr=info['lr']))
        print(result)

        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch', 'Batch Time', 'Data Time', 'Loss', 'Prec@1', 'lr']

    if mode == 'test':
        result = (
            'Batch Time {batch_time} '
            'Epoch Time {epoch_time} \n'
            'Loss {loss} '
            'Prec@1 {top1} '.format(batch_time=info['Batch Time'], epoch_time=info['Epoch Time'],
                                      loss=info['Loss'], top1=info['Prec@1']))
        print(result)
        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch', 'Batch Time', 'Epoch Time', 'Loss', 'Prec@1']

    if not os.path.isfile(filename):
        df.to_csv(filename, index=False, columns=column_names)
    else:  # else it exists so append without writing the header
        df.to_csv(filename, mode='a', header=False, index=False, columns=column_names)


if __name__ == '__main__':
    main()
