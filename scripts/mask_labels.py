import torch 
from argparse import Namespace
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
import os
import sys
import random
import numpy as np
import cv2
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


@click.command()
@click.option(
    "--dataset_dir",
    type=str,
    help="path to the dataset dir",
    default="../../artifacts/gen_pseudolabels_ds_target",
)

def main(dataset_dir):
    source_img_dir = os.path.join(dataset_dir, "images/rgb")
    out_dir = os.path.join(dataset_dir, "images/hue_masks")
    annotation_dir = os.path.join(dataset_dir, "annotations")
    anno_out_dir = os.path.join(dataset_dir, "masked_anno")
    os.mkdir(out_dir)
    os.mkdir(anno_out_dir)

    for source_img_name in tqdm(os.listdir(source_img_dir)):
        if any(substr in source_img_name for substr in [".png", ".JPG", ".jpg"]):

            mixed = Image.open(os.path.join(source_img_dir, source_img_name))

            mixed = mixed.filter(ImageFilter.BLUR)
            hue_mask = np.array(mixed.convert('HSV'))[:,:,0]
            hue_mask[hue_mask < 40]=0
            hue_mask[hue_mask > 140]=0
            hue_mask[hue_mask > 0 ]=255

            hue_pil = Image.fromarray(hue_mask)
            hue_pil = hue_pil.filter(ImageFilter.MaxFilter(5))
            hue_pil = hue_pil.filter(ImageFilter.MinFilter(5))

            hue_pil.save(os.path.join(out_dir, f"{source_img_name.split('.')[0]}_huemask.png"))

            hue_np = np.array(hue_pil)
            hue_np[hue_np == 255] = 1
            hue_np[hue_np == 0] = 0.75
            masked = np.moveaxis(np.array(mixed), -1, 0) * hue_np
            masked = np.moveaxis(masked, 0, 2)
            masked_pil = Image.fromarray(masked)
            masked_pil.save(os.path.join(out_dir, f"{source_img_name.split('.')[0]}_masked.png"))

            mixed.save(os.path.join(out_dir, f"{source_img_name.split('.')[0]}_ori.png"))

            anno_pil = Image.open(os.path.join(annotation_dir, source_img_name))
            # anno_pil.filter(ImageFilter.MaxFilter(5))
            anno_old = np.array(anno_pil)
            anno_new = np.array(hue_pil)
            anno_new[anno_new==255] = 2  # all vege as weeds unless specified otherwise 
            anno_new +=1
            anno_new[anno_old == 1] -= 1
            anno_new[anno_new < 2] = 0
            anno_new[anno_new == 2] = 1
            anno_new[anno_new == 3] = 2

            bnw = np.array(hue_pil)

            # connected components to remove boundary issue
            output = cv2.connectedComponentsWithStats(bnw, 4, cv2.CV_32S)
            (num_labels, labels, stats, centroids) = output
            for l in range(num_labels):
                if len(np.unique(anno_new[labels == l])) > 1:
                    if (anno_new[labels == l]==1).sum() >  (anno_new[labels == l]==2).sum():
                        # TODO assumed there is no soil 
                        anno_new[labels == l] = 1
                    else:
                        anno_new[labels == l] = 2
                    
            anno_new_pil = Image.fromarray(anno_new)
            anno_new_pil.save(os.path.join(anno_out_dir, source_img_name))
            
            """ for visualisation only
            annos_vis = np.copy(anno_new)
            annos_vis[annos_vis == 1] = 255 
            annos_vis[annos_vis == 2] = 125 
            anno_vis_pip = Image.fromarray(annos_vis)

            anno_vis_pip.save(os.path.join(out_dir, f"{source_img_name.split('.')[0]}_annos_new_new_new.png")) # TODO: make a vis to see how the annos have changed
           """

        else:
            print(f"skipping file {source_img_name} because it is not an img")

if __name__ == '__main__':
    main()


