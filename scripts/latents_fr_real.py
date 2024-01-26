#!/usr/bin/env python3  
"""Generate latents and fake images from real images 

"""


import os 
import sys

from PIL import Image
import torch
from tqdm import tqdm
from argparse import Namespace
from torch.utils.data import DataLoader
import numpy as np 
import time 
import click

sys.path.append("../pixel2style2pixel")
sys.path.append("../pixel2style2pixel/scripts")

from models.psp import pSp
from utils.common import tensor2im
from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from sklearn import mixture


def save_gmm(gmm, path):
    os.makedirs(path, exist_ok=True)
    mean_path = os.path.join(path, "gmm_mean.npy")
    np.save(mean_path, gmm.means_, allow_pickle=False)

    cov_path = os.path.join(path, "gmm_cov.npy")
    np.save(cov_path, gmm.covariances_, allow_pickle=False)
    
    weights_path = os.path.join(path, "gmm_weights.npy")
    np.save(weights_path, gmm.weights_, allow_pickle=False)

    cholesky_path = os.path.join(path, "gmm_cholesky.npy")
    np.save(cholesky_path, gmm.precisions_cholesky_, allow_pickle=False)


def load_gmm(path):
    """
    https://gist.github.com/Kukanani/619a27d8a8cc1b245ef2d30f671a4a37
    """
    # reload
    # TODO OOP and modularize path retrivals
    mean_path = os.path.join(path, "gmm_mean.npy")
    cov_path = os.path.join(path, "gmm_cov.npy") 
    weights_path = os.path.join(path, "gmm_weights.npy")
    cholesky_path = os.path.join(path, "gmm_cholesky.npy")

    means = np.load(mean_path)
    covar = np.load(cov_path)
    
    loaded_gmm = mixture.GaussianMixture(n_components = len(means), covariance_type="diag") 

    loaded_gmm.precisions_cholesky_ = np.load(cholesky_path)
    loaded_gmm.weights_ = np.load(weights_path)
    loaded_gmm.means_ = means
    loaded_gmm.covariances_ = covar

    return loaded_gmm

def get_psp(f_en_ckpt):
    # load psp stuff
    ckpt = torch.load(f_en_ckpt, map_location='cpu')
    opts = ckpt['opts']
    opts['output_size'] = 512
    opts['checkpoint_path'] = f_en_ckpt
    opts = Namespace(**opts)
    psp_net = pSp(opts)
    psp_net.eval()
    psp_net.cuda()
    return psp_net, opts



def est_gmm(n_comps, sefa_tensor, out_dir, f_en_ckpt, data_path, test_workers):
    out_img_dir = os.path.join(out_dir, "plants")
    out_latent_dir = os.path.join(out_dir, "latents")
    out_source_dir = os.path.join(out_dir, "ori")
    out_multi_dir = os.path.join(out_dir, "multi")

    os.mkdir(out_dir)
    os.mkdir(out_img_dir)
    os.mkdir(out_latent_dir)
    os.mkdir(out_source_dir)
    os.mkdir(out_multi_dir)

    psp_net, opts = get_psp(f_en_ckpt)

    psp_latent_arr = None
    gmm = None

    print('Loading dataset for {}'.format(opts.dataset_type))
    print(f'Dataset path {data_path}')
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=test_workers,
                        drop_last=True)

    for count, input_batch in enumerate(tqdm(dataloader)):
        img_name = f"{count}.png"
        out_img_path = os.path.join(out_img_dir, img_name)
        out_latent_path = os.path.join(out_latent_dir, f"{count}.pt")
        
        input_cuda = input_batch.cuda().float()
        result_batch, psp_latent_pt  = psp_net(input_cuda, return_latents=True, resize=False)

        torch.save(psp_latent_pt, out_latent_path)
        out_pil = tensor2im(result_batch[0])
        out_pil.save(out_img_path)

        # save the source img for dbg 
        in_pil = tensor2im(input_batch[0])
        in_pil.save(os.path.join(out_source_dir, img_name))

        if psp_latent_arr is None:
            psp_latent_arr = psp_latent_pt.detach().cpu()
        else:
            psp_latent_arr = torch.cat([psp_latent_arr, psp_latent_pt.detach().cpu()])

    # fit latents 
    psp_latent_arr = torch.matmul(psp_latent_arr, sefa_tensor['eigvec'])
    latentPLUS_size = 16*512
    X = psp_latent_arr.reshape((psp_latent_arr.shape[0], latentPLUS_size)) 
    
    gmm = mixture.GaussianMixture(n_components=n_comps, covariance_type="diag")
    gmm.fit(X)
    max_latents = X.max(dim=0).values
    min_latents = X.min(dim=0).values

    return gmm, psp_net, max_latents, min_latents 


def get_vegmask(im_pil):
    hue_mask = np.array(im_pil.convert('HSV'))[:,:,0]
    hue_mask[hue_mask < 40]=0
    hue_mask[hue_mask > 140]=0
    hue_mask[hue_mask > 0 ]=1

    vm_score = hue_mask.sum()

    return vm_score


def sample_gmm(gmm, out_dir, num_samples_per_comp, sefa_tensor, psp_net):
    print("generating samples from gmm...")
    MIN_VEGMASK = 5000.

    out_multi_dir = os.path.join(out_dir, "multi") 
    os.makedirs(out_multi_dir, exist_ok=True)
    os.makedirs(os.path.join(out_multi_dir, "latents"), exist_ok=True)
    os.makedirs(os.path.join(out_multi_dir, "images"), exist_ok=True)
    for dist in range(1):
        start_time = time.time()
        samples_w_ = gmm.sample(num_samples_per_comp)

        print("sampling done. time taken in seconds:")
        print(time.time() - start_time)

        samples_w = samples_w_[0].reshape(num_samples_per_comp, 16, 512)

        for instance_count, sample_w in tqdm(enumerate(samples_w)): 
            sample_w = np.expand_dims(sample_w, axis=0)
            sefa_np = (sefa_tensor['eigvec'].T).numpy()
            sample_w = np.matmul(sample_w, sefa_np)

            # generate the images from the sampled w's 
            tensor_w = torch.from_numpy(sample_w).to(device='cuda').float() 
            generated, latent_ = psp_net.decoder([tensor_w], 
                input_is_latent=True,
                return_latents= True,)

            im_pil =  tensor2im(generated[0])

            veg_mask = get_vegmask(im_pil)

            if veg_mask > MIN_VEGMASK:
                im_pil.save(f"{out_multi_dir}/images/source_{dist}_{instance_count}_{veg_mask}.png")
                torch.save(latent_, f"{out_multi_dir}/latents/source_{dist}_{instance_count}_{veg_mask}.pt")


@click.command()
@click.option(
    "--sefa_path",
    type=str,
    help="path to the sefa factors",
    default="../../artifacts/sefa.pt",
)
@click.option(
    "--data_path",
    type=str,
    help="path to the rgb data folder",
    default="../../downloads/source_dataset/patched_512",
)
@click.option(
    "--gmm_path",
    type=str,
    help="path to dir with gmm params",
    default="../../artifacts/gmm",
)
@click.option(
    '--is_load_gmm', 
    is_flag=True,
    help='set this flag if you want to use the gmms at gmm_path. Otherwise, I will save gmms to gmm_path.'
)
@click.option(
    "--f_en_checkpoint_path",
    type=str,
    help="path to F_EN checkpoint",
    default="../../downloads/inverter_weights.pt",
)
@click.option(
    "--out_dir_path",
    type=str,
    help="path to the output folder. Should NOT exist beforehand",
    default="../../artifacts/source_generated_dataset",
)
@click.option(
    "--test_workers",
    type=int,
    help="number of dataloader cpu threads",
    default=1,
)

def main(sefa_path, data_path, gmm_path, is_load_gmm, f_en_checkpoint_path, out_dir_path, test_workers):
    n_comps = 100
    num_samples_per_comp = 15000

    sefa_tensor = torch.load(sefa_path)

    if is_load_gmm:
        gmm=load_gmm(gmm_path)
        psp_net , opts = get_psp(f_en_checkpoint_path) 
    else:
        gmm, psp_net, max_latents, min_latents = est_gmm(
                                                n_comps, 
                                                sefa_tensor, 
                                                out_dir_path,
                                                f_en_checkpoint_path,
                                                data_path,
                                                test_workers
                                                )
        save_gmm(gmm, gmm_path)
    
    sample_gmm(gmm, out_dir_path, num_samples_per_comp, sefa_tensor, psp_net)


if __name__ == '__main__':
    main()
