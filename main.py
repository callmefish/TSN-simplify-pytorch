import os
import torch.optim as optim
from dataset.dataset import TSNDataSet
from model.model import TSN
from dataset.transforms import *
from config import parser
from train import train, validate, test
import utils
import torch.backends.cudnn as cudnn

best_prec1 = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
#fake dataset to make sure the whole net archetecture 
#is correct before to run on GPUs

import torch.utils.data as data
class mydataset(data.Dataset):
    def __init__(self):
        super(mydataset, self).__init__()
    def __len__(self):
        return 4
    def __getitem__(self,index):
        return torch.rand(9,224,224),1
'''


def main():
    global args, best_prec1
    args = parser.parse_args()

    if not os.path.exists('./record'):
        os.mkdir('./record')

    cudnn.benchmark = True

    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400
    elif args.dataset == 'sthsth':
        num_class = 2
    else:
        raise ValueError('Unknown dataset' + args.dataset)

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch, consensus_type=args.consensus_type,
                dropout=args.dropout, partial_bn=not args.nopartial_bn)

    model = model.cuda()
    #model = torch.nn.DataParallel(model).cuda()


    # from torchsummary import summary
    # summary(model, input_size=(54, 224, 224))

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()  # 包括宽高比抖动、水平翻转
    start_epoch = 0

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint {} at epoch {}"). \
                  format(args.resume, start_epoch + 1))
        else:
            print("=> no checkpoint found at {}".format(args.resume))

    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    else:
        data_length = 5

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   On_Video=args.On_Video,
                   interval=args.interval,
                   image_tmpl="frame_{:06d}.jpg" if args.modality in ["RGB", "RGBDiff"] else "{}/{}/frame{:06d}.jpg",
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   On_Video=args.On_Video,
                   interval=args.interval,
                   image_tmpl="frame_{:06d}.jpg" if args.modality in ["RGB", "RGBDiff"] else "{}/{}/frame{:06d}.jpg",
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    '''
    train_loader = torch.utils.data.DataLoader(
        mydataset(),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        mydataset(),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    '''

    criterion = torch.nn.CrossEntropyLoss().cuda()

    for group in policies:
        print('group: {} has {} params lr_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult']))

    optimizer = optim.SGD(policies, args.lr, momentum=0.9, weight_decay=args.weight_decay)

    if args.evaluate:
        test(val_loader, model, num_class)
        return

    best_epoch = start_epoch
    for epoch in range(start_epoch, args.epochs):

        print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        train(args.record_path, train_loader, model, criterion, optimizer, epoch, args.clip_gradient)
        prec1 = validate(args.record_path,val_loader, model, criterion, epoch)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if is_best:
            filename = args.record_path + args.modality + 'best.pth'
            best_epoch = epoch
        else:
            filename = args.record_path + args.modality + str(epoch) + '.pth'

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1
        }, filename)
        if epoch - best_epoch > 10:
            return
        elif epoch - best_epoch > 5:
            print('epoch {} best epoch{}'.format(epoch + 1, best_epoch + 1))
            args.lr = utils.adjust_learning_rate(args.lr, optimizer)


if __name__ == '__main__':
    main()

    # import cv2
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # cap = cv2.VideoCapture('/home/qx/project/data/sthsth/data/123022.webm')
    #
    # frame_counts = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print('count frames ', frame_counts)
    # imglist = []
    # for i in range(frame_counts):
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    #     res, frame = cap.read()
    #
    #     if res:
    #         imglist.append(frame)
    #         cv2.imshow("capture", frame)
    #     else:
    #         print(i, res)
    #
    #     if cv2.waitKey(50) & 0xFF == ord('q'):
    #         break
    # imglist = np.array(imglist[::8])
    # print(imglist.shape)
    # imglist = imglist.transpose((1, 0, 2, 3)).reshape(240, -1, 3)
    # plt.imshow(imglist)
    # plt.show()
    #
    #
