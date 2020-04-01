# RobustPathFollow
Pytorch Implementation of "Visual Memory for Robust Path Following", NIPS18
* Success rate: 83.3%
* SPL: 75.2%

Trained with over 40000 episodes and tested with over 1000 episodes of habitat matterport dataset.


### Prerequisites
  - A basic Pytorch installation. I used pytorch 1.3.1
  - tensorboardX installation.
  - Habitat simulator ([Habitat sim](https://github.com/facebookresearch/habitat-sim)).
  
 
### Installation
1. Clone the repository
  ```Shell
  git clone https://github.com/bareblackfoot/RobustPathFollow.git
  ```

### Setup data
1. Download habitat-sim pathfollow dataset
    - Google drive [here](https://drive.google.com/drive/folders/1-XwH9nZkKDynqN227LvxjUVPhjv6BPzu?usp=sharing) 

2. Create a folder and a soft link to use the dataset
  ```Shell
  mkdir data
  cd data
  ln -s path/to/downloaded/dataset/pathfollow .
  ```

### Test with pre-trained models
1. Download pre-trained model
  - Google drive [here](https://drive.google.com/file/d/1Qd9FOAYf82kyUBezeg5aKA4e5Hp3jJB8/view?usp=sharing).
 
2. Locate the model inside the outputs/rpf_nuri
  ```Shell
  mv path/to/best.pth ./outputs/rpf_nuri
  ```
3. Test with pre-trained rpf models
  ```Shell
  GPU_ID=0
  CUDA_VISIBLE_DEVICES={GPU_ID} python eval.py
  ```

### Train your own model
1. Download pre-trained models and weights. The current code support VGG16 and Resnet V1 models. Pre-trained models are provided by slim, you can get the pre-trained models [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models) and set them in the ``data/imagenet_weights`` folder. For example for VGG16 model, you can set up like:
  ```Shell
  GPU_ID=0
  CUDA_VISIBLE_DEVICES={GPU_ID} python train.py
  ```

By default, trained networks are saved under:

```
outputs/default/
```

Test outputs are saved under:

```
outputs/default/
```

Tensorboard information for train and validation is saved under:

```
experiments/tb_logs/default/
```
