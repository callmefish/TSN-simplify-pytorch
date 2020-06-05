# #!/bin/bash
# #replace the variables with your github repo url, repo name, test
# #video name, json named by your UIN
# GIT_REPO_URL="https://github.com/callmefish/TSN-simplify-pytorch.git"
# REPO="TSN-simplify-pytorch"
# VIDEO="./sample_video/sample_video_04(S).mp4"
# UIN_JSON="529005218.json"
# UIN_JPG="529005218.jpg"
# git clone $GIT_REPO_URL
# cd $REPO
# gsutil cp gs://ucf101_for_rar/sample_video.zip ./
# unzip sample_video.zip
# gsutil cp gs://ucf101_for_rar/475_inceptionv4__flow_model_best.pth.tar ./
# gsutil cp gs://ucf101_for_rar/475_inceptionv4_rgb_model_best.pth.tar ./
# #Replace this line with commands for running your test python file.
# echo $VIDEO
# python test.py --video_name $VIDEO
# #If your test file is ipython file, uncomment the following lines and
# #replace IPYTHON_NAME with your test ipython file.
# #IPYTHON_NAME="test.ipynb"
# #echo $IPYTHON_NAME
# #jupyter notebook
# #rename the generated timeLabel.json and figure with your UIN.
# cp timeLable.json $UIN_JSON
# cp timeLable.jpg $UIN_JPG


#!/bin/bash
#replace the variables with your github repo url, repo name, test
GIT_REPO_URL="https://github.com/callmefish/TSN-simplify-pytorch.git"
REPO="TSN-simplify-pytorch"
VIDEO_File="./sample_video/"
UIN="529005218"
git clone $GIT_REPO_URL
cd $REPO
gsutil cp gs://ucf101_for_rar/sample_video.zip ./
unzip sample_video.zip
gsutil cp gs://ucf101_for_rar/668_resnet101_flow_model_best.pth.tar ./
gsutil cp gs://ucf101_for_rar/668_resnet101_rgb_model_best.pth.tar ./
echo $VIDEO
python win_test.py --video_file_name $VIDEO_File

# time label and probability picture are in the file 'result'   
# their names contain UIN
# video YouTube link:
# 1)  notslip 01: https://www.youtube.com/watch?v=PeH6aCMhdN4
# 2)  notslip 02: https://www.youtube.com/watch?v=9yUsQ3WBW4I  
# 3)  notslip 03: https://www.youtube.com/watch?v=gx40I8RDIvE 
# 4)  notslip 04: https://www.youtube.com/watch?v=bn-X360_Ae8
# 5)  notslip 05: https://www.youtube.com/watch?v=7mDiULkVBiU
# 6)  notslip 06: https://www.youtube.com/watch?v=aD5hYu3RXOk
# 7)  slip 01: https://www.youtube.com/watch?v=1vaIcYT9wNI  
# 8)  slip 02: https://www.youtube.com/watch?v=sV2YTSRabzc
# 9)  slip 03: https://www.youtube.com/watch?v=2gcLii-0HxY
# 10)  slip 04: https://www.youtube.com/watch?v=I9bJAAvtZ2g
# 11)  slip 05: https://www.youtube.com/watch?v=JjP-QoFeuUU 
# 12)  slip 06: https://www.youtube.com/watch?v=wFQJOvAeTqU
