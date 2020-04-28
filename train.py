from torch.nn.utils import clip_grad_norm
from utils import *
import time
import torch
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(path, train_loader, model, criterion, optimizer, epoch, clip_gradient=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()

    end = time.time()
    progress = tqdm(train_loader)
    for i, (data, target) in enumerate(progress):
        data_time.update(time.time() - end)
        target = target.cuda()
        data = data.cuda()
        output = model(data)
        loss = criterion(output, target)
        prec1 = accuracy(output.data, target, topk=(1, ))
        losses.update(loss.data, data.size(0))
        top1.update(prec1[0], data.size(0))

        optimizer.zero_grad()
        loss.backward()

        if clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), clip_gradient)
            if total_norm > clip_gradient:
                print("clipping gradient{} with coef {}".\
                      format(total_norm, clip_gradient / total_norm))
        optimizer.step()

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
    record_info(info, path+'train.csv', 'train')


def validate(path,val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()

    end = time.time()
    progress = tqdm(val_loader)
    with torch.no_grad():
        for i, (data, target) in enumerate(progress):
            data_time.update(time.time() - end)
            target = target.cuda()
            data = data.cuda()
            output = model(data)
            loss = criterion(output, target)
            prec1 = accuracy(output.data, target, topk=(1,))
            losses.update(loss.data, data.size(0))
            top1.update(prec1[0], data.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

    info = {'Epoch': [epoch],
            'Batch Time': [round(batch_time.avg, 3)],
            'Epoch Time': [round(batch_time.sum, 3)],
            'Data Time': [round(data_time.avg, 3)],
            'Loss': [round(losses.avg.item(), 5)],
            'Prec@1': [round(top1.avg.item(), 4)],
            }
    if path is not None:
        record_info(info, path+'test.csv', 'test')
    else:
        print(info)
    return top1.avg

def test(val_loader, model, num_classes):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    model.eval()

    end = time.time()

    class_num = np.array([0] * num_classes)
    class_prec1 = np.array([0] * num_classes)
    progress = tqdm(val_loader)
    with torch.no_grad():
        for i, (data, target) in enumerate(progress):
            data_time.update(time.time() - end)
            target = target.cuda()
            data = data.cuda()
            output = model(data)

            #def class_accuracy(outputs, targets, num_classes, topk=(1,), ):
            prec1, class_prec1_t, class_num_t  \
                = class_accuracy(output.data, target, num_classes, topk=(1, ))

            class_num += class_num_t
            class_prec1 += class_prec1_t

            top1.update(prec1[0], data.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

    info = {
            'Batch Time': [round(batch_time.avg, 3)],
            'Epoch Time': [round(batch_time.sum, 3)],
            'Data Time': [round(data_time.avg, 3)],
            'Prec@1': [round(top1.avg.item(), 4)],
            }

    print(info)


    print(np.argsort(class_prec1/class_num))
    print(np.sort(class_prec1/class_num))

