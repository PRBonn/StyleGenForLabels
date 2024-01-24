import torch 
from argparse import Namespace
from PIL import Image
import torchvision.transforms as transforms
import os
import sys
import random
import numpy as np

sys.path.append(".")
sys.path.append("..")
from options.test_options import TestOptions
from models.psp import pSp
from configs import data_configs
from utils.common import tensor2im
from models.encoders import psp_encoders
from models.psp import get_keys
from models.stylegan2.model import Generator



latent_np_path = "/media/linn/export4tb/cache/chongral22_files/target/it460k/sampled_ugv_average_latent.npy"
latent_np = np.load(latent_np_path)
latent_torch = torch.from_numpy(latent_np).to(device="cuda:0")


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



def one_pair(opts, dummy_annos, source_latent_path, target_latent_path, trunc, tag=None, alpha=1.0):
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

    # swap_w = latent_torch 
    swap_w = swap_w * alpha + latent_torch * (1.0-alpha)

    # my_noise = [None] * 16
    # my_noise = torch.rand(1,1,4, device=opts.device)
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


if __name__ == '__main__':
    test_opts = TestOptions().parse()
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)
    opts.unpaired=False

    stylegan_ckpt = "/mnt/exp13/ckpts/sg2_ckpts/batch-mixed-labelled/checkpts/460000.pt"
    # stylegan_ckpt = "/mnt/exp13/ckpts/sg2_ckpts/batch-mixed-labelled/checkpts/440000.pt"

    """template
    source_latent_dir=""
    source_img_dir=""
    target_img_dir=""
    target_latent_dir=""
    out_dir=""
    """

    # truncation trick for revision
    source_latent_dir="/media/linn/export4tb/cache/chongral22_files/source/it460k/multi/latents"
    source_img_dir="/media/linn/export4tb/cache/chongral22_files/source/it460k/multi/imgs"
    target_img_dir="/media/linn/export4tb/cache/chongral22_files/target/it460k/multi/imgs"
    target_latent_dir="/media/linn/export4tb/cache/chongral22_files/target/it460k/multi/latents"
    out_dir="/media/linn/export4tb/cache/chongral22_files/jan16_trunc0_5"

    """
    # no filtering (for gan eval)
    source_latent_dir="/media/linn/export4tb/cache/chongral22_files/source/it460k_maxmin_all/multi/latents"
    source_img_dir="/media/linn/export4tb/cache/chongral22_files/source/it460k_maxmin_all/multi/imgs"
    target_img_dir="/media/linn/export4tb/cache/chongral22_files/target/it460k_maxmin_all/multi/imgs"
    target_latent_dir="/media/linn/export4tb/cache/chongral22_files/target/it460k_maxmin_all/multi/latents"
    out_dir="/media/linn/export4tb/cache/chongral22_files/maxmin_gan"
    """

    # # feb3 one2one
    # source_latent_dir="/media/linn/export4tb/cache/chongral22_files/source/it460k/latents"
    # source_img_dir="/media/linn/export4tb/cache/chongral22_files/source/it460k/plants"
    # target_img_dir="/media/linn/export4tb/cache/chongral22_files/target/it460k/plants"
    # target_latent_dir="/media/linn/export4tb/cache/chongral22_files/target/it460k/latents"
    # out_dir="/media/linn/export4tb/cache/chongral22_files/feb3_one2one"

    # # dec 21 it460k
    # source_latent_dir="/media/linn/export4tb/cache/chongral22_files/source/it460k/multi/latents"
    # source_img_dir="/media/linn/export4tb/cache/chongral22_files/source/it460k/multi/imgs"
    # target_img_dir="/media/linn/export4tb/cache/chongral22_files/target/it460k/multi/imgs"
    # target_latent_dir="/media/linn/export4tb/cache/chongral22_files/target/it460k/multi/latents"
    # out_dir="/media/linn/export4tb/cache/chongral22_files/jan16_vm3"



    """nov 
    source_latent_dir = "/media/linn/export4tb/cache/chongral22_files/onetoone/latents"
    source_img_dir = "/media/linn/export4tb/cache/chongral22_files/onetoone/ori"  # FIXME i only need this because the classification uav/ugv is done only in img dir
    target_img_dir = "/media/linn/export4tb/cache/chongral22_files/target/onetoone/ori"
    target_latent_dir = "/media/linn/export4tb/cache/chongral22_files/target/onetoone/latents"
    out_dir = "/media/linn/export4tb/cache/chongral22_files/target/onetoone/mixing"
    """

    """ dec 15, one-to-one case 
    source_latent_dir = "/media/linn/export4tb/cache/chongral22_files/onetoone/latents"
    source_img_dir = "/media/linn/export4tb/cache/chongral22_files/onetoone/plants"
    target_img_dir = "/media/linn/export4tb/cache/chongral22_files/target/vm_scored_nov29/plants_vm5000"
    target_latent_dir = "/media/linn/export4tb/cache/chongral22_files/target/vm_scored_nov29/latents_vm5000"
    out_dir = "/media/linn/export4tb/cache/chongral22_files/target/vm_scored_nov29/mixed_1-1"
    """

    """
    # dec 16 sampled k100 
    source_latent_dir="/media/linn/export4tb/cache/chongral22_files/source/vm_scored/multi/latents_4k"
    source_img_dir="/media/linn/export4tb/cache/chongral22_files/source/vm_scored/multi/imgs_4k"
    target_img_dir="/media/linn/export4tb/cache/chongral22_files/target/vm_scored/multi/imgs_5k"
    target_latent_dir="/media/linn/export4tb/cache/chongral22_files/target/vm_scored/multi/latents_5k"
    out_dir="/media/linn/export4tb/cache/chongral22_files/mixed/uav4k_ugv5k"
    """


    net = Generator(
        512, 512, 8, channel_multiplier=2
    ).to(opts.device)
    checkpoint = torch.load(stylegan_ckpt)

    net.load_state_dict(checkpoint["g_ema"])
 
    net.eval()
    net.cuda()


    # ckpt_dir = "/mnt/exp13/ckpts/psp_ckpts/running/bm_pt_ds_mixed_y_39_tv_l10_mse-pt"
    # style_en_ckpt = "/mnt/exp13/ckpts/psp_ckpts/running/en_pt_batch-mixed_uni30_mse/checkpoints/iteration_240000.pt"
    swap_pos_onwards = 7
    
    dummy_annos = "/media/linn/7ABF-E20F1/da_data/UAVbonn2017/train_only/annotations_onehot_styles/f12/sugar_f3_170930_02_subImages_2_frame90_crop4.png"
    
    # out_dir = os.path.join(f"/mnt/exp13/ckpts/sg2_ckpts/batch-mixed-labelled/stylemixing_latents_ugv_norand_baseline{swap_pos_onwards}-Lonly-test")
    os.mkdir(out_dir)

    # target_recon_dir = os.path.join(out_dir, "target_recon")
    ori_source_dir = os.path.join(out_dir, "source_ori")
    ori_target_dir = os.path.join(out_dir, "target_ori")
    mixed_dir = os.path.join(out_dir, "mixed")
    os.mkdir(mixed_dir)
    os.mkdir(ori_source_dir)
    os.mkdir(ori_target_dir)

    """
    print(f"reading style encoder from {style_en_ckpt}...")
    style_ckpt = torch.load(style_en_ckpt)
    opts.n_styles = 16
    style_encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', opts)
    style_encoder.load_state_dict(get_keys(style_ckpt, 'encoder'), strict=True)
    style_encoder.cuda().eval()
    """

    for source_img_name in os.listdir(source_img_dir):
        if any(substr in source_img_name for substr in [".png", ".JPG", ".jpg"]):
            source_latent_name = source_img_name.split('.')[0]+".pt"
            source_latent_path = os.path.join(source_latent_dir, source_latent_name)
            target_img_name = random.choice(os.listdir(target_img_dir))
            target_latent_path = os.path.join(target_latent_dir, target_img_name.split('.')[0]+".pt")
            # for tag in range(10):
            tag = None
            if True:
                mixed, source_ori, target_ori= one_pair(opts, dummy_annos, source_latent_path, target_latent_path, swap_pos_onwards, tag=tag)

                if tag is not None:
                    source_img_namev = source_img_name.split('.')[0]+f"_v{tag}.png"
                else:
                    source_img_namev = source_img_name.split('.')[0]+".png"
                mixed.save(os.path.join(mixed_dir, source_img_namev))

                source_ori.save(os.path.join(ori_source_dir, source_img_namev))
                target_ori.save(os.path.join(ori_target_dir, f"{source_img_namev}_{target_img_name}"))
        else:
            print(f"skipping file {source_img_name} because it is not an img")


