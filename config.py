import argparse

parser = argparse.ArgumentParser(description="Pytorch impelementation of  Temporal Segment Networks")

parser.add_argument('--dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics','sthsth'], default='sthsth')
parser.add_argument('--modality', type=str, choices=['RGB', 'RGBDiff', 'Flow'], default='RGB')
parser.add_argument('--train_list', type=str, default='C:/Users/yzy97/Documents/master/csce636/TSN-pytorch/TSN-pytorch-master/train_list.txt')
parser.add_argument('--val_list', type=str, default='C:/Users/yzy97/Documents/master/csce636/TSN-pytorch/TSN-pytorch-master/test_list.txt')
parser.add_argument('--root_path', type=str, default='C:/Users/yzy97/Documents/untitled/video/video_data/video_data_475/')
parser.add_argument('--On_Video', type=bool, default=False)
parser.add_argument('--interval', type=int, default=2)
parser.add_argument('--record_path', type=str, default='record/')

# =======================model config==========================#
parser.add_argument('--arch', type=str, default='resnet101')
parser.add_argument('--num_segments', type=int, default=3)
parser.add_argument('--consensus_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk', 'identity', 'rnn', 'cnn']
                    )
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.5)

# =======================learning config==========================#
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--lr', default=0.005, type=float)
parser.add_argument('--clip-gradient', default=None, type=float)
parser.add_argument('--nopartial_bn', default=True, action="store_false")
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)

# ========================= Runtime Configs ==========================
parser.add_argument('--workers', default=1, type=int)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--evaluate', default=False, action='store_true')
parser.add_argument('--snapshot_pref', type=str, default="")
# parser.add_argument('--gpus', nargs='+', type=int, default=1)
