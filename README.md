# TSN-simplify-pytorch
Binary action recognition simplified from TSN-pytorch

# Run
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
 2. Download dataset and unzip them:\
	[rgb_475](https://storage.cloud.google.com/ucf101_for_rar/video_data_475.zip?authuser=1)(gs://ucf101_for_rar/video_data_475.zip): A dataset of RGB frames extracted from 475 action videos\
	[opt_475](https://storage.cloud.google.com/ucf101_for_rar/opt_475.zip?authuser=1)(gs://ucf101_for_rar/opt_475.zip): A dataset of TVL1 optical flow frames extracted from 475 action videos.\
	[rgb_575](https://storage.cloud.google.com/ucf101_for_rar/video_data_575.zip?authuser=1) (gs://ucf101_for_rar/video_data_575.zip)(optional): A dataset of RGB frames extracted from 475 action videos.\
	[opt_575](https://storage.cloud.google.com/ucf101_for_rar/opt_575.zip?authuser=1) (gs://ucf101_for_rar/opt_575.zip) (optional): A dataset of TVL1 optical flow frames extracted from 475 action videos.
  

 3. Download model and unzip:\
	[best_model_475](https://storage.cloud.google.com/ucf101_for_rar/opt_575.zip?authuser=1) (gs://ucf101_for_rar/best_model_475.zip): The spatial stream model and motion stream model trained from rgb_475 and opt_475 respectively.\
  

# Test model
Run "test_model.py". You can change any models.
