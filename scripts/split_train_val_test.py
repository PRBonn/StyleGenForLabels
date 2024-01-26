#!/usr/bin/env python3

import os
import pdb
from sklearn.model_selection import train_test_split
import click

@click.command()
@click.option(
    "--img_dir",
    type=str,
    help="path to all rgb images",
    default="../../artifacts/gen_pseudolabels_ds_target/images/rgb",
)
def main(img_dir):
    outfile = os.path.join(img_dir, "../../split.yaml")
    test_split = 0.15
    val_split = 0.15
    val_test_split = val_split / (1.0-test_split)

    imgs_l = os.listdir(img_dir)

    imgs_l_train, imgs_l_test, _ , _ = train_test_split(imgs_l, imgs_l, test_size=test_split, random_state=1993)
    imgs_l_train, imgs_l_val, _ , _ = train_test_split(imgs_l_train, imgs_l_train, test_size=val_test_split, random_state=1993)


    l_str_test ="\n- ".join(imgs_l_test)
    l_str_train ="\n- ".join(imgs_l_train)
    l_str_val ="\n- ".join(imgs_l_val)

    # write to file 
    fp = open(outfile, "w")
    fp.write("test:\n- ")
    fp.write(l_str_test)

    fp.write("\ntrain:\n- ")
    fp.write(l_str_train)

    fp.write("\nvalid:\n- ")
    fp.write(l_str_val)

if __name__ == '__main__':
    main()


