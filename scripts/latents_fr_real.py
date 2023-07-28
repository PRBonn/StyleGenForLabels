#!/usr/bin/env python3  
"""Generate latents and fake images from real images 

opts:

  checkpoint_path:  pSp checkpoint path 
  data_path:        path to real dir 
  

"""


import os 
import sys
sys.path.append("../pixel2style2pixel")
sys.path.append("../pixel2style2pixel/scripts")


from PIL import Image
import torch
from tqdm import tqdm
from argparse import Namespace
from torch.utils.data import DataLoader
import numpy as np 

sys.path.append("../pixel2style2pixel")
sys.path.append("../pixel2style2pixel/scripts")

from models.psp import pSp
from utils.common import tensor2im
from options.test_options import TestOptions
from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from sklearn import mixture
import time 

# TODO global variable bad.
COV_TYPE="diag"


def save_gmm(gmm, path):
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
    
    loaded_gmm = mixture.GaussianMixture(n_components = len(means), covariance_type=COV_TYPE) 

    loaded_gmm.precisions_cholesky_ = np.load(cholesky_path)
    loaded_gmm.weights_ = np.load(weights_path)
    loaded_gmm.means_ = means
    loaded_gmm.covariances_ = covar

    return loaded_gmm


def est_gmm(n_comps, sefa_tensor, is_fit=True):
    # TODO: expose params
    print("starting script...")
    out_dir = "./out_dir"

    out_img_dir = os.path.join(out_dir, "plants")
    out_latent_dir = os.path.join(out_dir, "latents")
    out_source_dir = os.path.join(out_dir, "ori")
    # FIXME terrible that this is here when the file saving is made outside this fn 
    out_multi_dir = os.path.join(out_dir, "multi")

    if is_fit:
        os.mkdir(out_img_dir)
        os.mkdir(out_latent_dir)
        os.mkdir(out_source_dir)
        os.mkdir(out_multi_dir)

    # load psp stuff
    test_opts = TestOptions().parse()    
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts['output_size'] = 512
    opts = Namespace(**opts)
    psp_net = pSp(opts)
    psp_net.eval()
    psp_net.cuda()

    psp_latent_arr = None
    gmm = None

    if is_fit:
        print('Loading dataset for {}'.format(opts.dataset_type))
        print(f'Dataset path {opts.data_path}')
        dataset_args = data_configs.DATASETS[opts.dataset_type]
        transforms_dict = dataset_args['transforms'](opts).get_transforms()
        dataset = InferenceDataset(root=opts.data_path,
                                   transform=transforms_dict['transform_inference'],
                                   opts=opts)
        dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
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

        # TODO plot the latents 
        # fit latents 
        psp_latent_arr = torch.matmul(psp_latent_arr, sefa_tensor['eigvec'])
        latentPLUS_size = 16*512
        X = psp_latent_arr.reshape((psp_latent_arr.shape[0], latentPLUS_size)) 
        torch.save(X, "X.pt")
        
        gmm = mixture.GaussianMixture(n_components=n_comps, covariance_type=COV_TYPE)
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


def sample_gmm(gmm, max_latents=None, min_latents=None):
    print("generating samples from gmm...")
    MIN_VEGMASK = 5000.

    # FIXME fml pls just make it an args
    out_multi_dir = "/media/linn/export4tb/cache/chongral22_files/source/it460k_all/multi"
    fail_dir = "/media/linn/export4tb/cache/chongral22_files/source/it460k_all/failed_vm"
    os.makedirs(fail_dir, exist_ok=True)

    for dist in range(1):
        start_time = time.time()
        samples_w_ = gmm.sample(num_samples_per_comp)

        print("sampling done")
        print(time.time() - start_time)

        samples_w = samples_w_[0].reshape(num_samples_per_comp, 16, 512)

        # TODO I am sure I can do this in a larger batch but you know, I am running out of fucks to give
        for instance_count, sample_w in enumerate(samples_w): 
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
                im_pil.save(f"{out_multi_dir}/source_{dist}_{instance_count}_{veg_mask}.png")
                torch.save(latent_, f"{out_multi_dir}/source_{dist}_{instance_count}_{veg_mask}.pt")
            else:
                im_pil.save(f"{fail_dir}/source_{dist}_{instance_count}_{veg_mask}.png")


   

if __name__ == '__main__':
    n_comps = 100
    num_samples_per_comp = 15000

    sefa_path = "./factor.pt"
    gmm_path="./gmm_output_dir"

    is_gen_gmm=True

    sefa_tensor = torch.load(sefa_path)
    gmm, psp_net, max_latents, min_latents = est_gmm(n_comps, sefa_tensor, is_fit=is_gen_gmm)

    if is_gen_gmm:
        save_gmm(gmm, gmm_path)
    else:
        gmm=load_gmm(gmm_path)
    
    sample_gmm(gmm, max_latents, min_latents)
