import os

first_path = 'C:/Users/yzy97/Documents/untitled/video/video_data/video_data_475/'
first_path_sub = os.listdir(first_path)
first_path_sub.sort()

with open('test_list.txt', 'w') as f:
    for i in range(len(first_path_sub)):
        if (i + 1) % 4 == 0:
            video_name = first_path_sub[i]
            video_class = first_path_sub[i].split('_')[1]
            class_num = 0 if video_class == 'Slipping' else 1
            frame = len(os.listdir(first_path + video_name + '/'))
            f.write("{:s}\t\t{:d}\t{:d}\n".format(video_class + '/' + video_name, frame, class_num))
    f.close()

with open('train_list.txt', 'w') as f:
    for i in range(len(first_path_sub)):
        if (i + 1) % 4 > 0:
            video_name = first_path_sub[i]
            video_class = first_path_sub[i].split('_')[1]
            class_num = 0 if video_class == 'Slipping' else 1
            frame = len(os.listdir(first_path + video_name + '/'))
            f.write("{:s}\t\t{:d}\t{:d}\n".format(video_class + '/' + video_name, frame, class_num))
    f.close()
