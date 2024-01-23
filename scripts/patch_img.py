#!/usr/bin/env python3

import os
from torchvision import transforms
from PIL import Image
from glob import glob
import numpy as np
import math
import click
from tqdm import tqdm


def one_dir(img_dir, out_parent_dir, patch_w, patch_h, overlap, resize_r):
    out_dir = os.path.join(out_parent_dir,os.path.basename(img_dir))

    os.mkdir(out_dir)

    imgs=os.listdir(img_dir)
    for img_name in tqdm(imgs):
        imgpath = os.path.join(img_dir, img_name)
        dirpath = os.path.join(out_dir, img_name.split('.')[0])
        os.mkdir(dirpath)

        if (".JPG" in imgpath) or (".png" in imgpath) or (".jpg" in imgpath):  
            img_pil = Image.open(imgpath)
            ori_width, ori_height = img_pil.size

            img_tensor = transforms.ToTensor()(img_pil)
            img_tensor = transforms.Resize(size=(int(ori_height*resize_r), int(ori_width*resize_r)))(img_tensor)
            img_np = img_tensor.permute(1, 2, 0).numpy()
            ori_height = img_np.shape[0]
            ori_width = img_np.shape[1]
            count = 0 
            top_col_x = 0

            while top_col_x + patch_w <= ori_width:
                top_row_y = 0 

                while top_row_y + patch_h <= ori_height:
                    patch_name=img_name.split('.')[0]+"_"+str(count)+"_"+str(top_row_y)+"_"+str(top_col_x)+".png"
                    cropped_img = img_np[top_row_y:top_row_y+patch_h, top_col_x:top_col_x+patch_w]
                    cropped_img_pil = Image.fromarray(cropped_img.astype(np.uint8))
                    cropped_img_pil.save(os.path.join(dirpath, patch_name))
                    top_row_y = top_row_y + patch_h - overlap
                    count+=1
                top_col_x = top_col_x + patch_w - overlap
        else:
            print("skipping file", imgpath)

@click.command()
@click.option(
    "--img_dirs",
    type=str,
    help="glob path for img dir(s)",
    default="../../downloads/source_dataset/rgb",
)
@click.option(
    "--out_dir",
    type=str,
    help="path for output dir",
    default="../../downloads/source_dataset/patched_512",
)
def main(img_dirs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    patch_w = 512
    patch_h = 512
    overlap = 0
    resize_r = 1.0

    for img_dir in glob(img_dirs, recursive = False):
        print(f"patching for imgs in {img_dir}")
        one_dir(img_dir, out_dir, patch_w, patch_h, overlap, resize_r)


if __name__ == "__main__":
    main()

