# StyleGenForLabels

This repo contains our research code used in the [paper](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/chong2023ral.pdf) "Unsupervised Generation of Labeled Training Images for Crop-Weed Segmentation in New Fields and on Different Robotic Platforms". 
The code in this repo has been frozen at the time of submission of the paper and will not be extended, but feel free to email me if you have any issues.

<!-- TODO (Linn): maybe write a better description for those who do not read the paper -->
<!-- TODO (Linn): it would be cool to add shield badges but i dont think any are relevant rn -->

## Reproducing Paper's Results
### Datasets
Datasets used in the paper are available online:
+ [Source dataset (UAVBonn17) RGB images](https://www.ipb.uni-bonn.de/html/projects/chong2023ral/source_rgb_uavbonn17x3.zip)
+ [Target dataset (UGVBonn17) RGB images](https://www.ipb.uni-bonn.de/html/projects/chong2023ral/target_rgb_ugvbonn17.zip)
+ [Full generated train dataset](https://www.ipb.uni-bonn.de/html/projects/chong2023ral/generated_dataset.zip)
+ [(Unlabelled) large pretrained dataset](https://www.ipb.uni-bonn.de/html/projects/chong2023ral/bonn16ugv-all.zip)   
For the source and target datasets, we are still working on public access for the labels. Meanwhile, you can email me to get preliminary access.


### Weights
Weights of the following are also available online:
<!-- TODO  + If you want to train your own data: [pretrained StyleGAN2](https://www.ipb.uni-bonn.de/html/projects/chong2023ral/stylegan2_ugvbonn16_weights.pt) -->
+ [StyleGAN2 trained on source and target](https://www.ipb.uni-bonn.de/html/projects/chong2023ral/stylegan2_weights.pt)
+ [inverter](https://www.ipb.uni-bonn.de/html/projects/chong2023ral/inverter_weights.pt)
+ [Semantic segmentation](https://www.ipb.uni-bonn.de/html/projects/chong2023ral/erfnet_source.ckpt)

## Installation

### Requirements
+ We used the NVIDIA RTX A6000 GPU with CUDA version 11 for training our networks.
+ We used Python 3.6.7
+ Our system is a Linux machine. 

If you have a different setup, match the packages to your specific setup.

### Environment Setup
Clone this repo:
```sh
git clone --recurse-submodules https://github.com/PRBonn/StyleGenForLabels.git
cd StyleGenForLabels 
```
Then, use the requirements.txt to setup the python environment of your choosing via pip/pip3:
```sh
pip install -r requirements.txt
```
To install PyTorch, we needed to use specific wheels to support our GPU sm archi:
```sh
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
Or, you can install PyTorch directly if this is not a problem for you:
```sh
pip install torch==1.10.2 torchaudio==0.10.1
```


## Usage
### Training StyleGAN2 
You can use [StyleGAN2](https://www.ipb.uni-bonn.de/html/projects/chong2023ral/stylegan2_weights.pt) to pretrain your model.    
OR if you want to train your own model:
Follow the instructions from stylegan2-pytorch to train the StyleGAN2 on the source and target domain.   
Remember to use the [pretrained weights](https://www.ipb.uni-bonn.de/html/projects/chong2023ral/stylegan2_ugvbonn16_weights.pt) to train StyleGAN2 for your source and target domain (both domains should be mixed into a single lmdb)!    
<!-- TODO (Linn): 
(if you have two separate lmdb's for the source and target images separately, use the script to sample from each lmdb in each training batch).   
-->

### Training Encoder $F_{EN}$
1. Train the encoder:
```sh
cd pixel2style2pixel
```
2. Change the path in configs/paths_config.py to point to your dir with all the rgb images from source and test domain.
The directories should contain both source and target rgb images combined together into a single directory. 
```python
'plant_train': '../dataset/train_images/',
'plant_test': '../dataset/test_images/',
```

3. Run the training script
```sh
python scripts/train.py \ 
--dataset_type=plants_seg_to_real_uav \ 
--exp_dir=<output_dir_path> \
--workers=<number_of_workers> \
--batch_size=<training_batch_size> \
--test_batch_size=<test_batch_size> \
--test_workers=<number_of_test_workers> \
--val_interval=<steps_per_validation> \
--save_interval=<steps_per_checkpoint> \
--image_interval=<steps_per_sample_saving> \
--encoder_type=GradualStyleEncoder \
--label_nc=<number_of_input_channels> \
--input_nc=<number_of_output_channels> \
--output_size=<output_image_size> \
--max_steps=<max_number_of_steps_to_train_for> \
--stylegan_weights=<path_to_stylegan2_checkpoint> \
--learning_rate=<learning_rate>

```
For example:
```sh
python scripts/train.py \
--dataset_type=plants_seg_to_real_uav \
--exp_dir=../../F_EN_training \
--workers=0 \
--batch_size=16 \
--test_batch_size=16 \
--test_workers=0 \
--val_interval=5000 \
--save_interval=10000 \
--image_interval=1000 \
--encoder_type=GradualStyleEncoder \
--label_nc=3 \
--input_nc=3 \
--output_size=512 \
--max_steps=250000 \
--stylegan_weights=../../checkpts/stylegan2_weights.pt \
--learning_rate=0.0001
```

### Generating Training Image-Label Pairs
0. Calculate SEFA factors
```sh
cd stylegan2-pytorch
python closed_form_factorization.py  --out <output_bin_name.pt> <stylegan2_checkpoint>
```

#### 1. Generate Source images:
```sh
cd scripts
```
1. patch source images to 512
```sh
python patch_img.py --img_dirs <dir of input rgb> --out_dir <dir of output>
```
2. Generate source images and corresponding latents 
```sh
python latents_fr_real.py \
--sefa_path=<path_to_sefa_factors>
--data_path=<path_of_dir_with_source_images> \
--f_en_checkpoint_path=<path_of_F_EN_ckpt> \
--gmm_path=<path_to_where_gmm_will_be_saved> \
--out_dir_path=<path_to_where_the_generated_images_will_go> \
--test_workers=<number_of_cpu_threads_for_data_loading>
```

#### 2. Generate pseudo-labels:  

##### 2.1. Create the following dir structure with the source images and labels:
  + <parent_dir_of_dataset>
    + annotations
    + images
      + rgb   
    + split.yaml  # describes train-val-test split

##### 2.2. Train semantic segmentation on source image labels  
You can download the [semantic segmentation network weights here](https://www.ipb.uni-bonn.de/html/projects/chong2023ral/erfnet_source.ckpt)   
OR train your own network:
```sh
cd semantic_segmentation
vi ./config/config_train.yaml  # change the path_to_dataset to point ot your source dataset
python train.py --config ./config/config_train.yaml --export_dir <output_dir>
```


##### 2.3. Use said semantic segmentation network to generate pseudo labels for generated source images 
1. create dataset from generated source images (multi/images --> images/rgb; leave annotations dir empty)

2. change the config_pred.yaml to point to the generated source dataset 

3. train with the generated source dataset
```sh
cd semantic_segmentation
python predict.py --export_dir <output_dir> --config config/config_pred.yaml --ckpt_path erfnet_source.ckpt
```
4. copy the labels to the dataset dir (lightning_logs/version_XX/visualize/semantic_labels --> annotations)


#### 3. Generating Target images (style mixing):
```sh
cd scripts

python patch_img.py --img_dirs <dir of input rgb> --out_dir <dir of output>

python latents_fr_real.py \
--sefa_path=<path_to_sefa_factors>
--data_path=<path_of_dir_with_source_images> \
--f_en_checkpoint_path=<path_of_F_EN_ckpt> \
--gmm_path=<path_to_where_gmm_will_be_saved> \
--out_dir_path=<path_to_where_the_generated_images_will_go> \
--test_workers=<number_of_cpu_threads_for_data_loading>

python pkls_only.py \
--stylegan_ckpt=<path to the stylegan2 checkpoint> \
--f_en_checkpoint_path=<path to F_EN checkpoint> \
--source_latent_dir=<path to dir containing latent codes of the generated source dataset> \
--source_img_dir=<path to dir containing rgb images of the generated source dataset> \
--target_latent_dir=<path to dir containing latent codes of the generated target dataset> \
--target_img_dir=<path to dir containing rgb images of the generated target dataset> \
--out_dir=<path to the output_dir. Should not exist beforehand.>
```


#### 4. Label Refinement:
1. create a dataset from the generated mixed images and the generated source labels
(mixed_dataset/mixed --> images/rgb; lightning_logs/version_XX/visualize/semantic_labels --> annotations)

```sh
cd scripts
python mask_labels.py \
--target_img_dir=<path_to_dir_with_generated_target_images> \
--labels=<path_to_dir_with_labels_to_refine> \
--output_dir=<output_dir_path>
```


### Training semantic segementation on generated data
```sh
cd semantic_segmentation
python train.py --config ... --export_dir ..
```

### Testing semantic segmentation on target data

```sh
cd semantic_segmentation
python test.py --config ... --export_dir ... --ckpt_path ...
```

## Support
Open an issue or email Linn Chong at linn.chong@uni-bonn.de if you need help with something.


## Citation
If you use this code for academic purposes, cite the [paper](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/chong2023ral.pdf):
```bibtex
@article{chong2023ral,
author = {Y.L. Chong and J. Weyler and P. Lottes and J. Behley and C. Stachniss},
title = {{Unsupervised Generation of Labeled Training Images for Crop-Weed Segmentation in New Fields and on Different Robotic Platforms}},
journal = ral,
volume = {8},
number = {8},
pages = {5259--5266},
year = 2023,
issn = {2377-3766},
doi = {10.1109/LRA.2023.3293356},
note = {accepted},
codeurl = {https://github.com/PRBonn/StyleGenForLabels}
}
```

