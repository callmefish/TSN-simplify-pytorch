# TSN-simplify-pytorch
Binary action recognition simplified from TSN-pytorch

# Run csce636 model test for test.video
please run the 'test.sh'\
there is a little change in it for better use


# Environment
```
VM: Google cloud deeplearning-vm
Framework: PyTorch-1.4
Based on: Debian GNU/Linux 9.11 (stretch) (GNU/Linux 4.9.0-11-amd64 x86_64\n)
Programming Language: Python 3.7 from Anaconda.
Machine type: n1-highmem-4 (4 vCPUs, 26 GB memory)
GPUs: 1 x NVIDIA Tesla K80
```

# Prepare Steps

 1. Clone code from github:
 	```
 	git clone https://github.com/callmefish/TSN-simplify-pytorch.git
 	```
 2. Download dataset and unzip them (if you don't want to train model, just ignore this step):\
	[rgb_475](https://storage.cloud.google.com/ucf101_for_rar/video_data_475.zip?authuser=1)(gs://ucf101_for_rar/video_data_475.zip): A dataset of RGB frames extracted from 475 action videos\
	[opt_475](https://storage.cloud.google.com/ucf101_for_rar/opt_475.zip?authuser=1)(gs://ucf101_for_rar/opt_475.zip): A dataset of TVL1 optical flow frames extracted from 475 action videos.\
	[rgb_575](https://storage.cloud.google.com/ucf101_for_rar/video_data_575.zip?authuser=1) (gs://ucf101_for_rar/video_data_575.zip)(optional): A dataset of RGB frames extracted from 475 action videos.\
	[opt_575](https://storage.cloud.google.com/ucf101_for_rar/opt_575.zip?authuser=1) (gs://ucf101_for_rar/opt_575.zip) (optional): A dataset of TVL1 optical flow frames extracted from 475 action videos.
  

 3. Download sample video:\
	[sample_video](https://storage.cloud.google.com/ucf101_for_rar/sample_video.zip?authuser=1) (gs://ucf101_for_rar/sample_video.zip): Six sample videos.\
 4. Download model:\
	[475_inceptionv4__flow_model_best.pth.tar](https://storage.cloud.google.com/ucf101_for_rar/475_inceptionv4__flow_model_best.pth.tar?authuser=1) (gs://ucf101_for_rar/475_inceptionv4__flow_model_best.pth.tar): Motion stream model.\
	[475_inceptionv4_rgb_model_best.pth.tar](https://storage.cloud.google.com/ucf101_for_rar/475_inceptionv4_rgb_model_best.pth.tar?authuser=1) (gs://ucf101_for_rar/475_inceptionv4_rgb_model_best.pth.tar): Spatial stream model.\

# Test model for your own
Run "test.py". You can change args in it.
