This folder contains the code and pretrained model for GaitRef: Gait Recognition with Refined Sequential Skeletons, the CVPR 2023 submission. We provide the python script with the processed data during submission, which we will replace with data-preprocessing python files in our final version. 

### Environment Setup

 * We have tested our code and model on a single NVIDIA 3090 gpu with Centos 8 as well as A40 gpu on Ubuntu 18.04, with python 3.7.13, CUDA 11.1, pytorch 1.8.1.

 * A suggestion for this is to use conda and create an environment as follows

conda create -n gaitref python=3.7.13
conda activate gaitref

 * After you create a python 3.7.13, please use the following command for installing required PyTorch

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

### Data Preparation and Pretrained Models

 * Since silhouettes and skeletons for CASIA-B are publicly available online (http://www.cbsr.ia.ac.cn/GaitDatasetB-silh.zip for silhouettes and https://github.com/tteepe/GaitGraph/releases/download/v0.1/data.zip for skeletons), for simplicity during reviewing stage, we directly provide the processed data we used along with the pretrained models in the following anonymous storage (https://mega.nz/folder/FeUnWbqa#3Pew51i7BWVftLkZ5pbaJQ). Please download both of the files and place them in the current directory. Unzip them with the commands below

tar -xzf CASIA-B-mix.tar.gz
tar -xzf pretrained.tar.gz

### Reproduce the results on CASIA-B dataset

 * To produce the numbers for GaitMix, please use the following command and replace the GPU id with the id you want (>6 GB memory available and please only use ONE gpu for the default config)

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 lib/main.py --cfgs ./config/gaitglmix.yaml --phase test

 * for GaitRef, please use the following

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 lib/main.py --cfgs ./config/gaitglref.yaml --phase test
