# StyleGenForLabels

This is the official repo for the [paper](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/chong2023ral.pdf) "Unsupervised Generation of Labeled Training Images for Crop-Weed Segmentation in New Fields and on Different Robotic Platforms"

This repo is still under construction. We are still working on making sure everything works well but till then, you can email me if you have any issues.

<!-- TODO (Linn): maybe write a better description for those who do not read the paper -->
<!-- TODO (Linn): it would be cool to add shield badges but i dont think any are relevant rn -->
WARNING: The code in this repo has been frozen at the time of submission of the paper and will NOT be maintained.

## Reproducing Paper's Results
<!-- TODO -->
### Datasets
Datasets used in the paper are available online:
+ [Source dataset (UAVBonn17)]()
+ [Target dataset (UGVBonn17)]()
+ [Full generated train dataset](https://www.ipb.uni-bonn.de/html/projects/chong2023ral/generated_dataset.zip).
+ [(Unlabelled) large pretrained dataset]()
For the source and target datasets, we are still working on public access for the labels. Meanwhile, you can email me to get preliminary access.


### Weights
Weights of the following are also available online:
+ [StyleGAN2]()
+ [inverter]()


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
cd stylegenforlabels
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
Follow the instructions from stylegan2-pytorch to train the StyleGAN2 on the source and target domain.

### Training Encoder $F_{EN}$
1. First, you need to get the latent variable **w** and the image that the StyleGAN2 generates with said **w**:
```sh
cd scripts
python generate_mit_latents.py \
--size <image_size_in_px> \
--sample 1 \
--pics <number_of_pairs> \
--ckpt <path_to_StyleGAN2_checkpoint> \
--output_path <output_dir_path>
```
For example,
```sh
python generate.py \
--size 256 \
--sample 1 \
--pics 500 \
--ckpt ../weights/460000.pt \
--output_path ./styleinversion_trainset
```

2. Train the encoder:
```sh
cd pixel2style2pixel
python scripts/train.py \ 
--dataset_type=plants \ 
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
--dataset_type=plants \
--exp_dir=./F_EN_training \
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
--stylegan_weights=/export/data/linn/ckpts/sg2_ckpts/batch-mixed-labelled/checkpts/460000.pt \
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
python latents_fr_real.py \
--data_path=<path_of_dir_with_source_images> \
--checkpoint_path=<path_of_F_EN_ckpt> \
--test_workers=<number_of_test_workers>
```

#### 2. Generate pseudo-labels:  

##### 2.1. Create a dir with the following dir structure:
  + <parent_dir_of_dataset>
    + annotations
    + images
      + rgb   

##### 2.2. Train semantic segmentation on source image labels   
```sh
cd semantic_segmentation
python train.py --config ... --export_dir ..
```

##### 2.3. Use said semantic segmentation network to generate pseudo labels for generated source images 
```sh
cd semantic_segmentation
python predict.py --export_dir /media/linn/export4tb/cache/chongral22_files/jan16_vm2/dataset/preds --config config/config_51.yaml --ckpt_path /media/linn/7ABF-E20F/jan/erfnet/version_13/checkpoints/UAVBonn_epoch=463_val_loss=0.1339.ckpt
```


#### 3. Generating Target images (style mixing):
```sh
cd scripts
python latents_fr_real.py --data_path=/media/linn/7ABF-E20F/da_data/UGV/labelled/Bonn_2017/train_only/images/rgb --checkpoint_path=/mnt/exp13/ckpts/psp_ckpts/running/en_pt_it460k/checkpoints/iteration_240000.pt --test_workers=0

python pkls_only.py \
--exp_dir=<output_dir_path> \
--checkpoint_path=<path_to_F_EN_checkpoint> \
--data_path=<path_to_dir_with_source_images> \
--test_batch_size=<test_batch_size> \
--test_workers=<num_of_workers> \
--n_images=<num_of_images_to_generate> \
--latent_mask=<dimensions_of_latent_to_swap>

```


#### 4. Label Refinement:
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

