import torch 
from argparse import Namespace
from PIL import Image
import torchvision.transforms as transforms
import os
import sys
import random
import numpy as np
import click
from tqdm import tqdm

sys.path.append("../pixel2style2pixel")
sys.path.append("../pixel2style2pixel/scripts")

from models.psp import pSp
from configs import data_configs
from utils.common import tensor2im
from models.encoders import psp_encoders
from models.psp import get_keys
from models.stylegan2.model import Generator


def dataset_getitem(from_path, to_path, target_transform, source_transform, is_test= True):
    from_im = Image.open(from_path)

    from_im = from_im.convert('L')
    # print(f"input shape {from_im.size}")
    to_im = Image.open(to_path).convert('RGB')
    # print(f"output shape {to_im.size}")


    crop_size = 512

    # perform random crop here instead 
    crop_params=transforms.RandomCrop.get_params(to_im, (crop_size,crop_size))


    if True:
        if is_test:
            to_im_1=transforms.CenterCrop(crop_size)(to_im)
        else:
            to_im_1 = transforms.functional.crop(to_im, crop_params[0], crop_params[1], crop_params[2], crop_params[3])
        to_im_1 = target_transform(to_im_1)
        # print(f"new output shape {to_im_1.size()}")

    if True:  #  FIXME breaks code!
        if is_test:
            from_im_1=transforms.CenterCrop(crop_size)(from_im)
        else:
            from_im_1 = transforms.functional.crop(from_im, crop_params[0], crop_params[1], crop_params[2], crop_params[3])
        from_im_1 = source_transform(from_im_1)
        # print(f"new input shape {from_im_1.size()}")
    else:
        from_im_1 = to_im

    return from_im_1, to_im_1



def one_pair(opts, source_latent_path, target_latent_path, trunc, net, tag=None):
    # prepare image to be input to psp encoder to get w+ 
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()

    source_latent = torch.load(source_latent_path)

    target_latent = torch.load(target_latent_path)

    target_recon_w = target_latent
    source_recon_w = source_latent

    swap_w = target_recon_w.clone()
    swap_w[:,:trunc,:]=source_recon_w[:,:trunc,:]


    # TRUNCATION TRICK 
    # swap_w = swap_w * alpha + latent_torch * (1.0-alpha)

    my_noise = None
    is_rand_z = True

    swaped_img_t = net([swap_w],
                     input_is_latent=True,
                     noise=my_noise,
                     randomize_noise=is_rand_z,
                    )
    swaped_img = tensor2im(swaped_img_t[0][0])

    target_img = net([target_recon_w],
                     input_is_latent=True,
                     noise=my_noise,
                    randomize_noise=is_rand_z,
                    )
    target_ori = tensor2im(target_img[0][0])

    source_ori = net([source_latent],
                     input_is_latent=True,
                     noise=my_noise,
                    randomize_noise=is_rand_z,
                    )
    source_ori = tensor2im(source_ori[0][0])
    return swaped_img, source_ori, target_ori

@click.command()
@click.option(
    "--stylegan_ckpt",
    type=str,
    help="path to the stylegan2 checkpoint",
    default="../../downloads/stylegan2_weights.pt",
)
@click.option(
    "--f_en_checkpoint_path",
    type=str,
    help="path to F_EN checkpoint",
    default="../../downloads/inverter_weights.pt",
)
@click.option(
    "--source_latent_dir",
    type=str,
    help="path to dir containing latent codes of the generated source dataset",
    default="../../artifacts/source_generated_dataset/multi/latents",
)
@click.option(
    "--source_img_dir",
    type=str,
    help="path to dir containing rgb images of the generated source dataset",
    default="../../artifacts/source_generated_dataset/multi/images",
)
@click.option(
    "--target_latent_dir",
    type=str,
    help="path to dir containing latent codes of the generated target dataset",
    default="../../artifacts/target_generated_dataset/multi/latents",
)
@click.option(
    "--target_img_dir",
    type=str,
    help="path to dir containing rgb images of the generated target dataset",
    default="../../artifacts/target_generated_dataset/multi/images",
)
@click.option(
    "--out_dir",
    type=str,
    help="path to the output_dir. Should not exist beforehand.",
    default="../../artifacts/mixed_dataset",
)
def main(stylegan_ckpt, 
        f_en_checkpoint_path, 
        source_latent_dir, 
        source_img_dir, 
        target_latent_dir, 
        target_img_dir, 
        out_dir):
    ckpt = torch.load(f_en_checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = f_en_checkpoint_path
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 512 
    opts = Namespace(**opts)
    opts.unpaired=False

    net = Generator(
        512, 512, 8, channel_multiplier=2
    ).to(opts.device)
    checkpoint = torch.load(stylegan_ckpt)

    net.load_state_dict(checkpoint["g_ema"])
 
    net.eval()
    net.cuda()

    swap_pos_onwards = 7
    
    os.mkdir(out_dir)

    ori_source_dir = os.path.join(out_dir, "source_ori")
    ori_target_dir = os.path.join(out_dir, "target_ori")
    mixed_dir = os.path.join(out_dir, "mixed")
    os.mkdir(mixed_dir)
    os.mkdir(ori_source_dir)
    os.mkdir(ori_target_dir)

    for source_img_name in tqdm(os.listdir(source_img_dir)):
        if any(substr in source_img_name for substr in [".png", ".JPG", ".jpg"]):
            source_latent_name = source_img_name.split('.')[0]+".pt"
            source_latent_path = os.path.join(source_latent_dir, source_latent_name)
            target_latent_name = random.choice(os.listdir(target_latent_dir))
            target_img_name = target_latent_name.split('.')[0]+".png"
            target_latent_path = os.path.join(target_latent_dir, target_latent_name)
            # for tag in range(10):
            tag = None
            if True:
                mixed, source_ori, target_ori= one_pair(opts, source_latent_path, target_latent_path, swap_pos_onwards, net, tag=tag)

                if tag is not None:
                    source_img_namev = source_img_name.split('.')[0]+f"_v{tag}.png"
                else:
                    source_img_namev = source_img_name.split('.')[0]+".png"
                mixed.save(os.path.join(mixed_dir, source_img_namev))

                source_ori.save(os.path.join(ori_source_dir, source_img_namev))
                target_ori.save(os.path.join(ori_target_dir, f"{source_img_namev}_{target_img_name}"))
        else:
            print(f"skipping file {source_img_name} because it is not an img")



if __name__ == '__main__':
    main()
