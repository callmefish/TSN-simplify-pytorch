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
	[video_data_668](https://storage.cloud.google.com/ucf101_for_rar/video_data_668.zip?authuser=1) (gs://ucf101_for_rar/video_data_575.zip)(optional): A dataset of RGB frames and TVL1 optical flow frames extracted from 668 action videos.
	
 

 3. Download sample video:\
	[sample_video](https://storage.cloud.google.com/ucf101_for_rar/sample_video.zip?authuser=1) (gs://ucf101_for_rar/sample_video.zip): twelve sample videos.
	
 4. Download model:\
	[668_resnet101_flow_model_best.pth.tar](https://storage.cloud.google.com/ucf101_for_rar/668_resnet101_flow_model_best.pth.tar?authuser=1) (gs://ucf101_for_rar/668_resnet101_flow_model_best.pth.tar): Motion stream model.\
	[668_resnet101_rgb_model_best.pth.tar](https://storage.cloud.google.com/ucf101_for_rar/668_resnet101_rgb_model_best.pth.tar?authuser=1) (gs://ucf101_for_rar/668_resnet101_rgb_model_best.pth.tar): Spatial stream model.

 5. Before training model:\
	If you want to train your own model, you need to revise some 'parser.add_argument', such as train_list, val_list, root_path and so on. If you only need to test real life videos, being focused on win-test.py is enough.

# Test model for your own
Run "test.py". You can change args in it.
