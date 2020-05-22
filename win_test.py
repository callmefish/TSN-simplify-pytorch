import argparse
import time
import os
import pickle
import shutil

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix

from dataset import TSNDataSet
from models import TSN
from transforms import *
from ops import ConsensusModule
import cv2
from glob import glob
import json
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")

parser.add_argument('--test_list', type=str, default='video_test_list.txt')
parser.add_argument('--flow_weights', type=str, default='570_resnet101_flow_model_best.pth.tar')
parser.add_argument('--rgb_weights', type=str, default='570_resnet101_rgb_model_best.pth.tar')
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--save_scores', type=str, default="record/")
parser.add_argument('--test_segments', type=int, default=7)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--video_file_name', type=str, default='sample_video/')

args = parser.parse_args()
rgb_whole_pred = {}
opf_whole_pred = {}


def main(rgb_net, opf_net, StartFrame):
    rgb = TEST(rgb_net, 'RGB', StartFrame)
    rgb.run()
    opf = TEST(opf_net, 'Flow', StartFrame)
    opf.run()


def rgb_model():
    net = TSN(2, 1, 'RGB',
              base_model=args.arch,
              consensus_type=args.crop_fusion_type,
              dropout=args.dropout)
    checkpoint = torch.load(args.rgb_weights)
    # print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
    net.load_state_dict(checkpoint['state_dict'])
    return net


def opf_model():
    net = TSN(2, 1, 'Flow',
              base_model=args.arch,
              consensus_type=args.crop_fusion_type,
              dropout=args.dropout)
    checkpoint = torch.load(args.flow_weights)
    # print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
    net.load_state_dict(checkpoint['state_dict'])
    return net


class TEST():
    def __init__(self, net, modality, StartFrame):
        self.net = net
        self.modality = modality
        self.StartFrame = StartFrame

    def run(self):
        if args.test_crops == 1:
            cropping = torchvision.transforms.Compose(
                [GroupScale(self.net.scale_size), GroupCenterCrop(self.net.input_size)])
        elif args.test_crops == 10:
            cropping = torchvision.transforms.Compose([GroupOverSample(self.net.input_size, self.net.scale_size)])
        else:
            raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

        if self.modality == 'RGB':
            root_path = 'record/test/temp_chunk/'
            test_list = 'rgb_video_test_list.txt'
        else:
            root_path = 'record/test/temp_opf/'
            test_list = 'opf_video_test_list.txt'
        data_set = TSNDataSet(root_path, test_list, num_segments=args.test_segments,
                              new_length=1 if self.modality == "RGB" else 5,
                              modality=self.modality,
                              image_tmpl="frame_{:06d}.jpg" if self.modality in ["RGB", "RGBDiff"] else "{:06d}.jpg",
                              test_mode=True,
                              transform=torchvision.transforms.Compose([
                                  cropping,
                                  Stack(roll=False),
                                  ToTorchFormatTensor(div=True),
                                  GroupNormalize(self.net.input_mean, self.net.input_std),
                              ]))
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.workers, pin_memory=True)
        self.net.cuda()
        self.net.eval()
        data_gen = enumerate(data_loader)
        output = []
        for i, (keys, data, label) in data_gen:
            a = data.chunk(args.test_segments, 1)
            res = []
            for j in a:
                rst = self.eval_video((i, j, label))
                res.append(rst)
            output.append((res, label[0]))
        if self.modality == 'RGB':
            rgb_whole_pred[str(StartFrame)] = np.mean(output[0][0], axis=0)
        else:
            opf_whole_pred[str(StartFrame)] = np.mean(output[0][0], axis=0)
        return

    def eval_video(self, video_data):
        i, data, label = video_data
        num_crop = args.test_crops

        if self.modality == 'RGB':
            length = 3
        elif self.modality == 'Flow':
            length = 10
        else:
            raise ValueError("Unknown modality " + self.modality)

        input_var = data.view(-1, length, data.size(2), data.size(3)).cuda()
        rst = self.net(input_var).data.cpu().numpy().copy()
        return rst.reshape((num_crop, 1, 2)).mean(axis=0)[0]


def cal_for_frames(video_path, video_name, flow_path):
    frames = glob(os.path.join(video_path, '*.jpg'))
    frames.sort()
    prev = cv2.UMat(cv2.imread(frames[0]))
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i, frame_curr in enumerate(frames[1:]):
        curr = cv2.UMat(cv2.imread(frame_curr))
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        prev = curr
        if not os.path.exists(os.path.join(flow_path, video_name + '_u')):
            os.mkdir(os.path.join(flow_path, video_name + '_u'))
        cv2.imwrite(os.path.join(flow_path, video_name + '_u', "{:06d}.jpg".format(i + 1)), tmp_flow[:, :, 0])
        if not os.path.exists(os.path.join(flow_path, video_name + '_v')):
            os.mkdir(os.path.join(flow_path, video_name + '_v'))
        cv2.imwrite(os.path.join(flow_path, video_name + '_v', "{:06d}.jpg".format(i + 1)), tmp_flow[:, :, 1])
    return


def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    flow = cv2.UMat.get(flow)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0
    return flow


def extract_flow(video_path, video_name, flow_path):
    cal_for_frames(video_path, video_name, flow_path)
    print('complete:' + flow_path + video_name)
    return


def make_sure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return


def change_txt(index):
    f_video_test = open('rgb_video_test_list.txt', 'w')
    f_video_test.write("{:s}\t\t{:d}\t{:d}\n".format('Slipping' + '/' + 'v_Slipping_g00_c00', index, 1))
    f_video_test.close()
    f_video_test = open('opf_video_test_list.txt', 'w')
    f_video_test.write("{:s}\t\t{:d}\t{:d}\n".format('Slipping' + '/' + 'v_Slipping_g00_c00', index - 1, 1))
    f_video_test.close()


def softmax(data):
    data_exp = np.exp(data)
    return data_exp / np.sum(data_exp)

def revise_order(path, isRGB):
    path_sub = os.listdir(path)
    path_sub.sort()
    if isRGB:
        for j in range(len(path_sub)):
            old_name = path + path_sub[j]
            new_name = path + 'frame_' + str(j+1).zfill(6) + '.jpg'
            os.rename(old_name, new_name)
    else:
        for j in range(len(path_sub)):
            old_name = path + path_sub[j]
            new_name = path + str(j+1).zfill(6) + '.jpg'
            os.rename(old_name, new_name)
    return

def save_fig(x, y, title, save_path):
    plt.figure()
    plt.plot(x, y, linewidth=2, color='lightskyblue')
    plt.xlabel('time/s')
    plt.ylabel('Slipping probability')
    plt.ylim(0, 1.1)
    plt.xlim(0, duration + 0.2)
    plt.title(title)
    plt.savefig(save_path)
    # plt.show()
    plt.close()
    return


if __name__ == '__main__':
    make_sure_dir('result/')
    path = args.video_file_name
    path_sub = os.listdir(path)
    path_sub.sort()
    for video_name in path_sub:
        make_sure_dir('record/')
        file_path = path + video_name
        video_title = file_path.split('/')[-1][:-4]
        print(video_title)
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            rate = cap.get(5)
            FrameNumber = cap.get(7)
            duration = FrameNumber / rate
            width = cap.get(3)
            height = cap.get(4)
        print(duration)
        make_sure_dir('record/')
        rgb_outPutDirName = 'record/temp_chunk/'
        opf_outPutDirName = 'record/temp_opf/'
        make_sure_dir(rgb_outPutDirName)
        make_sure_dir(opf_outPutDirName)

        spatial_model = rgb_model()
        motion_model = opf_model()
        index = 0
        while True:
            res, image = cap.read()
            if not res:
                print('not res , not image')
                break
            else:
                if width < height:
                    pad = int((height - width) // 2 + 1)
                    # image = image[150:, :]
                    image = cv2.copyMakeBorder(image, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
                image = cv2.resize(image, (342, 256))
                cv2.imwrite(rgb_outPutDirName + 'frame_' + str(index + 1).zfill(6) + '.jpg', image)
                index += 1
        print('extract rgb finished')
        extract_flow(rgb_outPutDirName, 'v_Slipping_g00_c00', opf_outPutDirName)
        cap.release()
        time_lable = {}
        make_sure_dir('record/test/')
        make_sure_dir('record/test/temp_chunk/')
        make_sure_dir('record/test/temp_chunk/v_Slipping_g00_c00/')
        make_sure_dir('record/test/temp_opf/')
        make_sure_dir('record/test/temp_opf/v_Slipping_g00_c00_u/')
        make_sure_dir('record/test/temp_opf/v_Slipping_g00_c00_v/')

        frame = 1
        for i in range(1, 31):
            shutil.copyfile(rgb_outPutDirName + 'frame_' + "{:06d}.jpg".format(i),
                            'record/test/temp_chunk/v_Slipping_g00_c00/' + 'frame_' + "{:06d}.jpg".format(i))
            shutil.copyfile(opf_outPutDirName + 'v_Slipping_g00_c00_u/' + "{:06d}.jpg".format(i),
                            'record/test/temp_opf/v_Slipping_g00_c00_u/' + "{:06d}.jpg".format(i))
            shutil.copyfile(opf_outPutDirName + 'v_Slipping_g00_c00_v/' + "{:06d}.jpg".format(i),
                            'record/test/temp_opf/v_Slipping_g00_c00_v/' + "{:06d}.jpg".format(i))
        frame = 31
        StartFrame = frame - 31
        change_txt(frame-1)
        main(spatial_model, motion_model, StartFrame)

        while frame + 10 < index:
            for i in range(1, 11):
                os.remove('record/test/temp_chunk/v_Slipping_g00_c00/' + 'frame_' + "{:06d}.jpg".format(i))
                shutil.copyfile(rgb_outPutDirName + 'frame_' + "{:06d}.jpg".format(frame),
                                'record/test/temp_chunk/v_Slipping_g00_c00/' + 'frame_' + "{:06d}.jpg".format(frame))
                os.remove('record/test/temp_opf/v_Slipping_g00_c00_u/' + "{:06d}.jpg".format(i))
                shutil.copyfile(opf_outPutDirName + 'v_Slipping_g00_c00_u/' + "{:06d}.jpg".format(frame),
                                'record/test/temp_opf/v_Slipping_g00_c00_u/' + "{:06d}.jpg".format(frame))
                os.remove('record/test/temp_opf/v_Slipping_g00_c00_v/' + "{:06d}.jpg".format(i))
                shutil.copyfile(opf_outPutDirName + 'v_Slipping_g00_c00_v/' + "{:06d}.jpg".format(frame),
                                'record/test/temp_opf/v_Slipping_g00_c00_v/' + "{:06d}.jpg".format(frame))
                frame += 1
            revise_order('record/test/temp_chunk/v_Slipping_g00_c00/', True)
            revise_order('record/test/temp_opf/v_Slipping_g00_c00_u/', False)
            revise_order('record/test/temp_opf/v_Slipping_g00_c00_v/', False)
            StartFrame += 10
            main(spatial_model, motion_model, StartFrame)
        fig_x, fig_y, fig_y_rgb, fig_y_opf = [], [], [], []
        for key in list(rgb_whole_pred.keys()):
            cur_time = float(key) / rate
            new_key = str(float('%.3f' % cur_time))
            new_value = softmax(rgb_whole_pred[key] + 1 * opf_whole_pred[key]).tolist()
            rgb_value = softmax(rgb_whole_pred[key]).tolist()
            opf_value = softmax(opf_whole_pred[key]).tolist()
            time_lable[new_key] = new_value[0]
            fig_x.append(cur_time)
            fig_y.append(new_value[0])
            fig_y_rgb.append(rgb_value[0])
            fig_y_opf.append(opf_value[0])

        json_str = json.dumps(time_lable)
        with open('result/529005218_' + video_title + '_' + 'timelabel.json', 'w') as json_file:
            json_file.write(json_str)
            json_file.close()

        point_num = len(fig_y_rgb)
        one_count = 0
        zero_count = 0
        for i in fig_y_rgb:
            if abs(i - 1) < 1e-4:
                one_count += 1
            if abs(i - 0) < 1e-4:
                zero_count += 1
        # if one_count/point_num > 0.9:
        #     time_lable = {str(float('%.3f'%(float(key)/rate))): softmax(opf_whole_pred[key]).tolist()[0] for key in
        #                   list(rgb_whole_pred.keys())}
        #     fig_y = fig_y_opf.copy()
        # elif zero_count / point_num > 0.9:
        #     time_lable = {str(float('%.3f' % (float(key) / rate))): softmax(rgb_whole_pred[key]).tolist()[0] for key in
        #                   list(rgb_whole_pred.keys())}
        #     fig_y = fig_y_rgb.copy()

        fig_x_1 = fig_x[:1]
        fig_y_1 = fig_y[:1]
        fig_y_rgb_1 = fig_y_rgb[:1]
        fig_y_opf_1 = fig_y_opf[:1]

        for i in range(1, len(fig_x)):
            fig_x_1.append(fig_x[i] - 0.001)
            fig_x_1.append(fig_x[i])
            fig_y_1.append(fig_y_1[-1])
            fig_y_1.append(fig_y[i])
            fig_y_rgb_1.append(fig_y_rgb_1[-1])
            fig_y_rgb_1.append(fig_y_rgb[i])
            fig_y_opf_1.append(fig_y_opf_1[-1])
            fig_y_opf_1.append(fig_y_opf[i])
        fig_x_1.append(duration)
        fig_y_1.append(fig_y[-1])
        fig_y_rgb_1.append(fig_y_rgb[-1])
        fig_y_opf_1.append(fig_y_opf[-1])

        save_fig(fig_x_1, fig_y_1, 'TSN', 'result/529005218_' + video_title + '_Part6.jpg')
        save_fig(fig_x_1, fig_y_rgb_1, 'TSN-RGB', 'result/529005218_' + video_title + '_rgb' + '_Part6.jpg')
        save_fig(fig_x_1, fig_y_opf_1, 'TSN-Flow', 'result/529005218_' + video_title + '_opf' + '_Part6.jpg')
        rgb_whole_pred = {}
        opf_whole_pred = {}
        shutil.rmtree('record/')
