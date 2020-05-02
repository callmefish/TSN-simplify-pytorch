import argparse
import time
import os
import pickle

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix

from dataset import TSNDataSet
from models import TSN
from transforms import *
from ops import ConsensusModule

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('--modality', type=str, default='RGBDiff', choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('--root_path', type=str, default='/home/yzy20161103/csce636_project/project/video_data_475/')
parser.add_argument('--test_list', type=str, default='/home/yzy20161103/tsn-pytorch-master/rgb_test_list.txt')
parser.add_argument('--weights', type=str, default='/home/yzy20161103/tsn-pytorch-master/record/rgbdiff/475_BNInception_rgbdiff_model_best.pth.tar')
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--save_scores', type=str, default="record/")
parser.add_argument('--test_segments', type=int, default=12)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--flow_prefix', type=str, default='')

args = parser.parse_args()


num_class = 2

net = TSN(num_class, 1, args.modality,
          base_model=args.arch,
          consensus_type=args.crop_fusion_type,
          dropout=args.dropout)

checkpoint = torch.load(args.weights)
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))


# for k, v in list(checkpoint['state_dict'].items()):
#     print(k)
#     break
# base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
net.load_state_dict(checkpoint['state_dict'])

if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(net.input_size),
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(net.input_size, net.scale_size)
    ])
else:
    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))


data_set = TSNDataSet(args.root_path, args.test_list, num_segments=args.test_segments,
                   new_length=1 if args.modality == "RGB" else 5,
                   modality=args.modality,
                   image_tmpl="frame_{:06d}.jpg" if args.modality in ['RGB', 'RGBDiff'] else "{:06d}.jpg",
                   test_mode=True,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       GroupNormalize(net.input_mean, net.input_std),
                   ]))
data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=1, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True)

#net = torch.nn.DataParallel(net).cuda()
print(net)
# net = net.cuda()
# net.eval()

# data_gen = enumerate(data_loader)

# total_num = len(data_loader.dataset)
# output = []


# def eval_video(video_data):
#     i, data, label = video_data
#     num_crop = args.test_crops

#     if args.modality == 'RGB':
#         length = 3
#     elif args.modality == 'Flow':
#         length = 10
#     elif args.modality == 'RGBDiff':
#         length = 18
#     else:
#         raise ValueError("Unknown modality "+args.modality)
    
#     input_var = data.view(-1, length, data.size(2), data.size(3)).cuda()
#     # input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)),
#     #                                     volatile=True)
#     rst = net(input_var).data.cpu().numpy().copy()
#     return i, rst.reshape((num_crop, args.test_segments, num_class)).mean(axis=0).reshape(
#         (args.test_segments, 1, num_class)
#     ), label[0]


# max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

# pred_dict = {}
# video_name = []
# for i, (keys, data, label) in data_gen:
#     if i >= max_num:
#         break
#     rst = eval_video((i, data, label))
#     output.append(rst[1:])
#     video_name.append(keys[0])

# for i in range(total_num):
#     pred_dict[video_name[i]] = np.mean(output[i][0], axis=0)

# with open(args.save_scores + args.modality.lower() + '/' + args.arch + '_' + args.modality.lower() + '_scores.pickle','wb') as f:
#     pickle.dump(pred_dict,f)
#     f.close()
# video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]

# video_labels = [x[1] for x in output]

# # [[TP, FN], [FP, TN]]
# cf = confusion_matrix(video_labels, video_pred).astype(float)

# # TP + FN and FP + TN
# cls_cnt = cf.sum(axis=1)
# # TP and TN
# cls_hit = np.diag(cf)
# # TPR(Recall) and TNR(Specificity)
# cls_acc = cls_hit / cls_cnt

# print(cls_acc)

# print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
