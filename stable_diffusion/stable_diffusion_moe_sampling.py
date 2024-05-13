import torch
from diffusers import UnCLIPScheduler, DDPMScheduler, StableUnCLIPPipeline, StableDiffusionPipeline
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
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import time

# Inspired from transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock

class SparseMoeBlock(nn.Module):
    def __init__(self, config, experts):
        super().__init__()
        self.hidden_dim = config["hidden_size"]
        self.num_experts = config["num_local_experts"]
        self.top_k = config["num_experts_per_tok"]
        self.out_dim = config.get("out_dim", self.hidden_dim)

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False, dtype=torch.float16)
        self.experts = nn.ModuleList([deepcopy(exp) for exp in experts])

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        batch_size, sequence_length, f_map_sz = hidden_states.shape
        hidden_states = hidden_states.view(-1, f_map_sz)
        # hidden_states = hidden_states.view(torch.float)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        _, selected_experts = torch.topk(
            router_logits.sum(dim=0, keepdim=True), self.top_k, dim=1
        )
        routing_weights = F.softmax(
            router_logits[:, selected_experts[0]], dim=1
        )

        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, self.out_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Loop over all available experts in the model and perform the computation on each expert
        for i, expert_idx in enumerate(selected_experts[0].tolist()):
            expert_layer = self.experts[expert_idx]

            current_hidden_states = routing_weights[:, i].view(
                batch_size * sequence_length, -1
            ) * expert_layer(hidden_states)

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states = final_hidden_states + current_hidden_states
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, self.out_dim
        )
        return final_hidden_states


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

    expert_stable_unclip_model_id = "stabilityai/stable-diffusion-2-1"

    expert_pipe = StableDiffusionPipeline.from_pretrained(expert_stable_unclip_model_id, torch_dtype=data_type)
    experts = [pipe, expert_pipe]

    #Replace attention projection and FF
    up_idx_start = 1
    up_idx_end = len(pipe.unet.up_blocks)
    down_idx_start = 0
    down_idx_end = len(pipe.unet.down_blocks) - 1
    num_experts=2
    num_experts_per_tok=1
    moe_layers="all"

    for i in range(down_idx_start, down_idx_end):
        for j in range(len(pipe.unet.down_blocks[i].attentions)):
            for k in range(
                    len(pipe.unet.down_blocks[i].attentions[j].transformer_blocks)
            ):
                if not moe_layers == "attn":
                    config = {
                        "hidden_size": next(
                            pipe.unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .ff.parameters()
                        ).size()[-1],
                        "num_experts_per_tok": num_experts_per_tok,
                        "num_local_experts": len(experts),
                    }
                    # FF Layers
                    layers = []
                    for l in range(len(experts)):
                        layers.append(
                            deepcopy(
                                experts[l]
                                .unet.down_blocks[i]
                                .attentions[j]
                                .transformer_blocks[k]
                                .ff
                            )
                        )
                    pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].ff = SparseMoeBlock(config, layers)
                if not moe_layers == "ff":
                    ## Attns
                    config = {
                        "hidden_size": pipe.unet.down_blocks[i]
                        .attentions[j]
                        .transformer_blocks[k]
                        .attn1.to_q.weight.size()[-1],
                        "num_experts_per_tok": num_experts_per_tok,
                        "num_local_experts": num_experts,
                    }
                    layers = []
                    for l in range(len(experts)):
                        layers.append(
                            deepcopy(
                                experts[l]
                                .unet.down_blocks[i]
                                .attentions[j]
                                .transformer_blocks[k]
                                .attn1.to_q
                            )
                        )
                    pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn1.to_q = SparseMoeBlock(config, layers)

                    layers = []
                    for l in range(len(experts)):
                        layers.append(
                            deepcopy(
                                experts[l]
                                .unet.down_blocks[i]
                                .attentions[j]
                                .transformer_blocks[k]
                                .attn1.to_k
                            )
                        )
                    pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn1.to_k = SparseMoeBlock(config, layers)

                    layers = []
                    for l in range(len(experts)):
                        layers.append(
                            deepcopy(
                                experts[l]
                                .unet.down_blocks[i]
                                .attentions[j]
                                .transformer_blocks[k]
                                .attn1.to_v
                            )
                        )
                    pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn1.to_v = SparseMoeBlock(config, layers)

                    config = {
                        "hidden_size": pipe.unet.down_blocks[i]
                        .attentions[j]
                        .transformer_blocks[k]
                        .attn2.to_q.weight.size()[-1],
                        "num_experts_per_tok": num_experts_per_tok,
                        "num_local_experts": len(experts),
                    }

                    layers = []
                    for l in range(len(experts)):
                        layers.append(
                            deepcopy(
                                experts[l]
                                .unet.down_blocks[i]
                                .attentions[j]
                                .transformer_blocks[k]
                                .attn2.to_q
                            )
                        )
                    pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn2.to_q = SparseMoeBlock(config, layers)

                    config = {
                        "hidden_size": pipe.unet.down_blocks[i]
                        .attentions[j]
                        .transformer_blocks[k]
                        .attn2.to_k.weight.size()[-1],
                        "num_experts_per_tok": num_experts_per_tok,
                        "num_local_experts": len(experts),
                        "out_dim": pipe.unet.down_blocks[i]
                        .attentions[j]
                        .transformer_blocks[k]
                        .attn2.to_k.weight.size()[0],
                    }
                    layers = []
                    for l in range(len(experts)):
                        layers.append(
                            deepcopy(
                                experts[l]
                                .unet.down_blocks[i]
                                .attentions[j]
                                .transformer_blocks[k]
                                .attn2.to_k
                            )
                        )
                    pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn2.to_k = SparseMoeBlock(config, layers)

                    config = {
                        "hidden_size": pipe.unet.down_blocks[i]
                        .attentions[j]
                        .transformer_blocks[k]
                        .attn2.to_v.weight.size()[-1],
                        "out_dim": pipe.unet.down_blocks[i]
                        .attentions[j]
                        .transformer_blocks[k]
                        .attn2.to_v.weight.size()[0],
                        "num_experts_per_tok": num_experts_per_tok,
                        "num_local_experts": len(experts),
                    }
                    layers = []
                    for l in range(len(experts)):
                        layers.append(
                            deepcopy(
                                experts[l]
                                .unet.down_blocks[i]
                                .attentions[j]
                                .transformer_blocks[k]
                                .attn2.to_v
                            )
                        )
                    pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn2.to_v = SparseMoeBlock(config, layers)

    for i in range(up_idx_start, up_idx_end):
        for j in range(len(pipe.unet.up_blocks[i].attentions)):
            for k in range(
                    len(pipe.unet.up_blocks[i].attentions[j].transformer_blocks)
            ):
                if not moe_layers == "attn":
                    config = {
                        "hidden_size": next(
                            pipe.unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .ff.parameters()
                        ).size()[-1],
                        "num_experts_per_tok": num_experts_per_tok,
                        "num_local_experts": len(experts),
                    }
                    # FF Layers
                    layers = []
                    for l in range(len(experts)):
                        layers.append(
                            deepcopy(
                                experts[l]
                                .unet.up_blocks[i]
                                .attentions[j]
                                .transformer_blocks[k]
                                .ff
                            )
                        )
                    pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].ff = SparseMoeBlock(config, layers)

                if not moe_layers == "ff":
                    # Attns
                    config = {
                        "hidden_size": pipe.unet.up_blocks[i]
                        .attentions[j]
                        .transformer_blocks[k]
                        .attn1.to_q.weight.size()[-1],
                        "num_experts_per_tok": num_experts_per_tok,
                        "num_local_experts": len(experts),
                    }

                    layers = []
                    for l in range(len(experts)):
                        layers.append(
                            deepcopy(
                                experts[l]
                                .unet.up_blocks[i]
                                .attentions[j]
                                .transformer_blocks[k]
                                .attn1.to_q
                            )
                        )

                    pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn1.to_q = SparseMoeBlock(config, layers)

                    config = {
                        "hidden_size": pipe.unet.up_blocks[i]
                        .attentions[j]
                        .transformer_blocks[k]
                        .attn1.to_k.weight.size()[-1],
                        "num_experts_per_tok": num_experts_per_tok,
                        "num_local_experts": len(experts),
                    }
                    layers = []

                    for l in range(len(experts)):
                        layers.append(
                            deepcopy(
                                experts[l]
                                .unet.up_blocks[i]
                                .attentions[j]
                                .transformer_blocks[k]
                                .attn1.to_k
                            )
                        )

                    pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn1.to_k = SparseMoeBlock(config, layers)

                    config = {
                        "hidden_size": pipe.unet.up_blocks[i]
                        .attentions[j]
                        .transformer_blocks[k]
                        .attn1.to_v.weight.size()[-1],
                        "num_experts_per_tok": num_experts_per_tok,
                        "num_local_experts": len(experts),
                    }
                    layers = []

                    for l in range(len(experts)):
                        layers.append(
                            deepcopy(
                                experts[l]
                                .unet.up_blocks[i]
                                .attentions[j]
                                .transformer_blocks[k]
                                .attn1.to_v
                            )
                        )

                    pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn1.to_v = SparseMoeBlock(config, layers)

                    config = {
                        "hidden_size": pipe.unet.up_blocks[i]
                        .attentions[j]
                        .transformer_blocks[k]
                        .attn2.to_q.weight.size()[-1],
                        "num_experts_per_tok": num_experts_per_tok,
                        "num_local_experts": len(experts),
                    }
                    layers = []

                    for l in range(len(experts)):
                        layers.append(
                            deepcopy(
                                experts[l]
                                .unet.up_blocks[i]
                                .attentions[j]
                                .transformer_blocks[k]
                                .attn2.to_q
                            )
                        )

                    pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn2.to_q = SparseMoeBlock(config, layers)

                    config = {
                        "hidden_size": pipe.unet.up_blocks[i]
                        .attentions[j]
                        .transformer_blocks[k]
                        .attn2.to_k.weight.size()[-1],
                        "out_dim": pipe.unet.up_blocks[i]
                        .attentions[j]
                        .transformer_blocks[k]
                        .attn2.to_k.weight.size()[0],
                        "num_experts_per_tok": num_experts_per_tok,
                        "num_local_experts": len(experts),
                    }

                    layers = []

                    for l in range(len(experts)):
                        layers.append(
                            deepcopy(
                                experts[l]
                                .unet.up_blocks[i]
                                .attentions[j]
                                .transformer_blocks[k]
                                .attn2.to_k
                            )
                        )

                    pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn2.to_k = SparseMoeBlock(config, layers)

                    config = {
                        "hidden_size": pipe.unet.up_blocks[i]
                        .attentions[j]
                        .transformer_blocks[k]
                        .attn2.to_v.weight.size()[-1],
                        "out_dim": pipe.unet.up_blocks[i]
                        .attentions[j]
                        .transformer_blocks[k]
                        .attn2.to_v.weight.size()[0],
                        "num_experts_per_tok": num_experts_per_tok,
                        "num_local_experts": len(experts),
                    }
                    layers = []

                    for l in range(len(experts)):
                        layers.append(
                            deepcopy(
                                experts[l]
                                .unet.up_blocks[i]
                                .attentions[j]
                                .transformer_blocks[k]
                                .attn2.to_v
                            )
                        )

                    pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn2.to_v = SparseMoeBlock(config, layers)
    
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
    print(exectime / 4)
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
    torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)
