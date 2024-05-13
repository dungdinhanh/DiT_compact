import torch
from diffusers import UnCLIPScheduler, DDPMScheduler, StableUnCLIPPipeline
from diffusers.models import PriorTransformer
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
import math
import argparse
import hfai
import hfai.multiprocessing

import hfai.client
from diffusion import create_diffusion, dist_util
import hfai.nccl.distributed as dist
from utils.handling_files import *
from datasets.coco_helper import load_data_caption, load_data_caption_hfai
import random
import time

def main(local_rank):
    """
    Run sampling
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

    # Create folder to save samples

    sample_folder_dir = os.path.join(args.sample_dir, f"images")
    prompt_folder_dir = os.path.join(args.sample_dir, f"prompts")
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        os.makedirs(prompt_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir} and .txt prompt at {prompt_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    list_png_files, max_index = get_png_files(sample_folder_dir)
    no_png_files = len(list_png_files)
    if no_png_files >= args.num_fid_samples:
        if rank == 0:
            print(f"Complete sampling {no_png_files} satisfying >= {args.num_fid_samples}")
            create_npz_from_sample_folder(args.sample_dir, args.num_fid_samples, args.image_size)
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
    total = max_index

    caption_loader = load_data_caption_hfai(split="val", batch_size=args.per_proc_batch_size)

    caption_iter = iter(caption_loader)
    prior_model_id = "kakaobrain/karlo-v1-alpha"
    data_type = torch.float16
    prior = PriorTransformer.from_pretrained(prior_model_id, subfolder="prior", torch_dtype=data_type)

    prior_text_model_id = "openai/clip-vit-large-patch14"
    prior_tokenizer = CLIPTokenizer.from_pretrained(prior_text_model_id)
    prior_text_model = CLIPTextModelWithProjection.from_pretrained(prior_text_model_id, torch_dtype=data_type)
    prior_scheduler = UnCLIPScheduler.from_pretrained(prior_model_id, subfolder="prior_scheduler")
    prior_scheduler = DDPMScheduler.from_config(prior_scheduler.config)

    stable_unclip_model_id = "stabilityai/stable-diffusion-2-1-unclip-small"

    pipe = StableUnCLIPPipeline.from_pretrained(
        stable_unclip_model_id,
        torch_dtype=data_type,
        # variant="fp16",
        prior_tokenizer=prior_tokenizer,
        prior_text_encoder=prior_text_model,
        prior=prior,
        prior_scheduler=prior_scheduler,

    )
    pytorch_total_params = sum(p.numel() for p in pipe.unet.parameters())
    print(pytorch_total_params)
    start = time.time()
    for _ in pbar:
        prompts = next(caption_iter)
        while len(prompts) != args.per_proc_batch_size:
            prompts = next(caption_iter)

        pipe = pipe.to(dist_util.dev())
        samples = pipe(prompt=prompts).images

        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            sample_np = center_crop_arr(sample, args.image_size)
            sample = numpy_to_pil(sample_np)[0]
            sample.save(f"{sample_folder_dir}/{index:06d}.png")
            f = open(f"{prompt_folder_dir}/{index:06d}.txt", "w")
            f.write(prompts[i])
            f.write("\n")
            f.close()
        total += global_batch_size
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(args.sample_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()
    exectime = time.time() - start
    print(exectime/4)
# image1, image2 = pipe(prompt=wave_prompt).images
# image1.save("test.png")
# image2.save("test2.png")

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=1)
    parser.add_argument("--num-fid-samples", type=int, default=30000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action='store_true', default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--fix_seed", action="store_true")
    return parser


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]

    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

if __name__ == "__main__":
    ngpus = torch.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=False)
