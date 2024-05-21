# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
# import hfai.nccl.distributed as dist
from models_compact import DiT_models
from download import find_model
from diffusion import create_diffusion, dist_util
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
import hfai
# import hfai.multiprocessing
from torch.utils import tensorboard
# import hfai.client
import time


from utils.handling_files import *

def main(local_rank):
    """
    Run sampling.
    """
    args = create_argparser().parse_args()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    # dist.init_process_group("nccl")
    dist_util.setup_dist(local_rank)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    if args.fix_seed:
        import random
        seed = args.global_seed * dist.get_world_size() + rank
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    else:
        print(f"Starting rank={rank}, random seed, world_size={dist.get_world_size()}.")

    torch.cuda.set_device(device)

    base_folder = args.base_folder
    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(os.path.join(base_folder, ckpt_path))
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}", cache_dir=base_folder).to(device)
    vae.decoder.eval()
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "models"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    output_folder_path = os.path.join(base_folder, args.sample_dir)
    sample_folder_dir = os.path.join(output_folder_path, f"images")
    reference_dir = os.path.join(output_folder_path, "reference")
    os.makedirs(reference_dir, exist_ok=True)
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    list_png_files, max_index = get_png_files(sample_folder_dir)

    final_file = os.path.join(reference_dir,
                              f"samples_{args.num_fid_samples}x{args.image_size}x{args.image_size}x3.npz")
    if os.path.isfile(final_file):
        dist.barrier()
        print("Sampling complete")
        dist.barrier()
        dist.destroy_process_group()
        return

    checkpoint = os.path.join(sample_folder_dir, "last_samples.npz")

    if os.path.isfile(checkpoint):
        all_images = list(np.load(checkpoint)['arr_0'])
    else:
        all_images = []
        all_images = compress_images_to_npz(sample_folder_dir, all_images)

    no_png_files = len(all_images)
    if no_png_files >= args.num_fid_samples:
        if rank == 0:
            print(f"Complete sampling {no_png_files} satisfying >= {args.num_fid_samples}")
            # create_npz_from_sample_folder(os.path.join(base_folder, args.sample_dir), args.num_fid_samples, args.image_size)
            arr = np.stack(all_images)
            arr = arr[: args.num_fid_samples]
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(reference_dir, f"samples_{shape_str}.npz")
            # logger.log(f"saving to {out_path}")
            print(f"Saving to {out_path}")
            np.savez(out_path, arr)
            os.remove(checkpoint)
            print("Done.")
        dist.barrier()
        dist.destroy_process_group()
        return
    else:
        if rank == 0:
            # remove_prev_npz(args.sample_dir, args.num_fid_samples, args.image_size)
            print("continue sampling")

    total_samples = int(math.ceil((args.num_fid_samples - no_png_files) / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Already sampled {no_png_files}")
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = len(all_images)

    skip = args.skip
    skip_type = args.skip_type
    respace_gap = int(1000/int(args.num_sampling_steps))
    if skip_type == "linear":
        guidance_timesteps = get_guidance_timesteps_linear(int(args.num_sampling_steps), skip)
    else:
        guidance_timesteps = get_guidance_timesteps_with_weight(int(args.num_sampling_steps), skip)


    current_samples = 0
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0) # remove this one could help boost performance/ in the models half = x instead of half = x[: len(x) // 2] + remove eps = torch.cat([half_eps, half_eps], dim=0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, guidance_steps=guidance_timesteps, respace_gap=respace_gap)
            sample_fn = model.forward_with_cfg_compact
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        # Sample images:
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples


        # samples = vae.decode(samples / 0.18215).sample
        samples = vae.decode(samples / 0.18215, return_dict=True)[0]

        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # if rank ==   0:
        #     if hfai.client.receive_suspend_command():
        #         if hfai.client.receive_suspend_command():
        #             print("Receive suspend - good luck next run ^^")
        #             hfai.client.go_suspend()
        dist.barrier()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size
        current_samples += global_batch_size
        dist.barrier()
        if current_samples >= 500 or total >= total_samples:
            if rank == 0:
                all_images = compress_images_to_npz(sample_folder_dir, all_images)
            current_samples = 0
            pass




    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        # create_npz_from_sample_folder(args.sample_dir, args.num_fid_samples)
        print(f"Complete sampling {total} satisfying >= {args.num_fid_samples}")
        # create_npz_from_sample_folder(os.path.join(base_folder, args.sample_dir), args.num_fid_samples, args.image_size)
        arr = np.stack(all_images)
        arr = arr[: args.num_fid_samples]
        shape_str = "x".join([str(x) for x in arr.shape])
        reference_dir = os.path.join(output_folder_path, "reference")
        os.makedirs(reference_dir, exist_ok=True)
        out_path = os.path.join(reference_dir, f"samples_{shape_str}.npz")
        # logger.log(f"saving to {out_path}")
        print(f"Saving to {out_path}")
        np.savez(out_path, arr)
        os.remove(checkpoint)
        print("Done.")
        # print("Done.")
    dist.barrier()
    dist.destroy_process_group()


def compress_images_to_npz(sample_folder_dir, all_images=[]):
    npz_file = os.path.join(sample_folder_dir, "last_samples.npz")
    list_png_files, _ = get_png_files(sample_folder_dir)
    no_png_files = len(list_png_files)
    if no_png_files <= 1:
        return all_images
    for i in range(no_png_files):
        image_png_path = os.path.join(sample_folder_dir, f"{list_png_files[i]}")
        try:
            # image_png_path = os.path.join(images_dir, f"{list_png_files[i]}")
            img = Image.open(image_png_path)
            img.verify()
        except(IOError, SyntaxError) as e:
            print(f'Bad file {image_png_path}')
            print(f'remove {image_png_path}')
            os.remove(image_png_path)
            continue
        sample_pil = Image.open(os.path.join(image_png_path))
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        all_images.append(sample_np)
        os.remove(image_png_path)
    np_all_images = np.stack(all_images)
    np.savez(npz_file, arr_0=np_all_images)
    return all_images

def get_guidance_timesteps_linear(n=250, skip=5):
    # T = n - 1
    # max_steps = int(T/skip)
    guidance_timesteps = np.zeros((n,), dtype=int)
    for i in range(n):
        timestep = i + 1
        if timestep % skip == 0:
            guidance_timesteps[i] = 1
        pass
    guidance_timesteps[0] = 1
    guidance_timesteps[1] = 1
    return guidance_timesteps
    pass


def get_guidance_timesteps_with_weight(n=250, skip=5):
    # c * i^2
    T = n - 1
    max_steps = int(n/skip)
    c = n/(max_steps**2)
    guidance_timesteps = np.zeros((n,), dtype=int)
    for i in range(max_steps):
        guidance_index = - int(c * (i ** 2)) + T
        if 0 <= guidance_index and guidance_index <= T:
            guidance_timesteps[guidance_index] += 1
        else:
            print(f"guidance index: {guidance_index}")
            print(f"constant c: {c}")
            print(f"faulty index: {i}")
            print(f"timesteps {T}")
            print(f"compressd by {skip} times")
            print(f"error in index must larger than 0 or less than {T}")
            exit(0)
    guidance_timesteps[1] = 1
    guidance_timesteps[0] = 1
    # print(guidance_timesteps)
    # print(np.sum(guidance_timesteps))
    # exit(0)
    return guidance_timesteps


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action='store_true', default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--fix_seed", action="store_true")
    parser.add_argument("--base_folder", type=str, default="./")
    parser.add_argument("--skip_type", type=str, choices=["linear", "quadratic"], default="linear")
    parser.add_argument("--skip", type=int, default=5)
    return parser


if __name__ == "__main__":
    ngpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)
