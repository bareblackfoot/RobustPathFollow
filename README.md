# RobustPathFollow
Pytorch Implementation of "[Visual Memory for Robust Path Following](https://papers.nips.cc/paper/7357-visual-memory-for-robust-path-following.pdf)" by Nuri Kim, Obin Kwon, and Hwiyeon Yoo.

### Results
- Our implementation:
    * Success rate: 86.0%
    * SPL: 77.3%
- In paper:
    * Success rate: 86.6%
    * SPL: 72.6%
    
We think the difference on performance came from the difference of the dataset. We collected the dataset with our own.
The RPF (Robust Path Following) model is trained with over 40000 episodes and tested with over 1000 episodes of habitat-sim pathfollow dataset.

### Prerequisites
  - A basic Pytorch installation. I used pytorch 1.3.1  
  - tensorboardX installation.
  - (For test) Habitat simulator ([Habitat sim](https://github.com/facebookresearch/habitat-sim)).
  
 
### Installation
1. Clone the repository
  ```Shell
  git clone https://github.com/bareblackfoot/RobustPathFollow.git
  ```


### Setup data
1. Download habitat-sim pathfollow test dataset
    - Google drive [here](https://drive.google.com/drive/folders/1-a3dU6oqNX4Hdu5HXTbUHGQQ1A0E51CM?usp=sharing) 
    - Email me to get training dataset
    
2. Create a folder and a soft link to use the dataset
  ```Shell
  mkdir data
  cd data
  ln -s path/to/downloaded/dataset/pathfollow .
  cd ..
  ```

3. Download [Habitat sim](https://github.com/facebookresearch/habitat-sim) scene dataset
    - The full Matterport3D (MP3D) dataset for use with Habitat can be downloaded using the official Matterport3D download script as follows: python download_mp.py --task habitat -o path/to/download/. You only need the habitat zip archive and not the entire Matterport3D dataset. Note that this download script requires python 2.7 to run.

4. Create a folder and a soft link to use the scene dataset
  ```Shell
  cd data
  ln -s path/to/downloaded/dataset/scene_dataset .
  cd ..
  ```

5. Make sure your directory structure looks like:
  * RobustPathFollow
    * data
        * pathfollow
            * train
                * demo
                * follower
            * test
                * demo
                * follower
            * valid
                * demo
                * follower
        * scene_datasets
            * mp3d
    * outputs
        * rpf_nuri
            * best.pth
   
   
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
1. You can train your model with the habitat-sim training dataset (Email me to get it!):
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
