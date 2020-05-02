import os

first_path = '/home/yzy20161103/csce636_project/project/video_data_475/'
first_path_sub = os.listdir(first_path)
first_path_sub.sort()

f_rgb_test = open('rgb_test_list.txt', 'w')
f_opf_test = open('opf_test_list.txt', 'w')
f_rgb_train = open('rgb_train_list.txt', 'w')
f_opf_train = open('opf_train_list.txt', 'w')
for i in range(len(first_path_sub)):
    if (i + 1) % 4 == 0:
        video_name = first_path_sub[i]
        video_class = first_path_sub[i].split('_')[1]
        class_num = 0 if video_class == 'Slipping' else 1
        frame = len(os.listdir(first_path + video_name + '/'))
        f_rgb_test.write("{:s}\t\t{:d}\t{:d}\n".format(video_class + '/' + video_name, frame, class_num))
        f_opf_test.write("{:s}\t\t{:d}\t{:d}\n".format(video_class + '/' + video_name, frame - 1, class_num))
    else:
        video_name = first_path_sub[i]
        video_class = first_path_sub[i].split('_')[1]
        class_num = 0 if video_class == 'Slipping' else 1
        frame = len(os.listdir(first_path + video_name + '/'))
        f_rgb_train.write("{:s}\t\t{:d}\t{:d}\n".format(video_class + '/' + video_name, frame, class_num))
        f_opf_train.write("{:s}\t\t{:d}\t{:d}\n".format(video_class + '/' + video_name, frame-1, class_num))

f_rgb_test.close()
f_opf_test.close()
f_rgb_train.close()
f_opf_train.close()
