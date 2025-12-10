import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import json
import wandb
import datetime as dt
from data.mixed_dataloader import load_cached_mm_dataset, cache_mixed_collate_fn
from model.models import *
from utils.utils import *
from utils.train_utils import WarmupConstantScheduler
from train import *


def setup_ddp():
    """
    Initialize distributed using env:// so torchrun provides RANK/WORLD_SIZE/LOCAL_RANK.
    Returns (rank, local_rank, world_size).
    """

    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=dt.timedelta(hours=1))
    rank = dist.get_rank()                       # global rank
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # per-node GPU index

    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True
    return rank, local_rank, world_size


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def main(global_rank, local_rank, world_size, args):
    # init (already done in setup_ddp)
    device_idx = local_rank                  # e.g., 0,1,2,...
    torch.cuda.set_device(device_idx)
    device = torch.device('cuda', device_idx)
    is_main_process = (global_rank == 0)

    n_epochs = args.n_epochs
    lr = args.lr
    exp_name = args.exp_name

    save_dir = f"ckpts/{exp_name}"
    args.save_every = 20000  # make this configurable later
    args.eval_every = 5000  # make this configurable later

    if is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        # Save config once
        with open(f"{save_dir}/config.json", "w") as f:
            json.dump(vars(args), f, indent=4)
        try:
            wandb.init(project="FlowBind", name=exp_name, config=vars(args))

        except Exception as e:
            print(f"[WARN] wandb init failed: {e}")

    models = {}

    ckpt = None

    text_model, image_model, audio_model = init_mlpflow(config=vars(args),
                                                        ckpt=ckpt,
                                                        **{mod: True for mod in args.modality},
                                                        eval_mode=False,
                                                        device=device,
                                                        verbose=is_main_process)

    # wrap with DDP
    for mod in args.modality:
        if mod == 'text':
            model = text_model
        elif mod == 'image':
            model = image_model
        elif mod == 'audio':
            model = audio_model
        else:
            raise ValueError(f"Unknown modality: {mod}")

        models[mod] = DDP(
            model, device_ids=[device], output_device=device_idx,
            find_unused_parameters=True, gradient_as_bucket_view=True
        )

    # --- Collect params after wrapping ---
    trainable_params = []
    for mod, ddp_model in models.items():

        cur_mod_param_cnt = 0
        for name, param in ddp_model.named_parameters():
            if not param.requires_grad:
                continue
            is_prior_param = 'logvar' in name
            is_flow_param = not is_prior_param

            trainable_params.append(param)
            cur_mod_param_cnt += param.numel()

        if is_main_process:
            print(f"Trainable params in {mod}: {cur_mod_param_cnt:,}")

    if is_main_process:
        total_params = sum(p.numel() for p in trainable_params)
        print(f"Total trainable params: {total_params:,}")

    dataset = load_cached_mm_dataset(
        dataset_list=args.dataset, cache_paths=args.cache_paths)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=global_rank, drop_last=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=cache_mixed_collate_fn
    )

    optimizer = optim.Adam(trainable_params, lr=lr)
    scheduler = WarmupConstantScheduler(
        optimizer, warmup_steps=args.warmup_steps) if hasattr(args, 'warmup_steps') else None

    global_step = 0

    # Only rank-0 needs the sample captions for logging
    log_img_captions = None

    # --- Train ---
    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)

        train_dict, global_step = train_one_epoch(
            global_rank, models, dataloader, optimizer, scheduler, device, args, epoch, global_step, log_img_captions
        )

        del train_dict

    print(f"Rank {global_rank}: Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=1001)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--exp_name", type=str, default="debug")

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--input_dim", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--num_layers", type=int, default=12)

    # path to resume the model
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        '--modality',
        nargs='+',
        type=str,
        help='Modality to use for training. e.g. text, image, audio',
        default=['text', 'image', 'audio']  # Default to all modalities
    )

    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of warmup steps for the learning rate scheduler", required=False)  # 1000

    # storetrue
    parser.add_argument("--var_clamp", type=float,
                        help="clamping constant the variance to a minimum value", default=0.0)
    parser.add_argument(
        '--dataset',
        nargs='+',
        type=str,
        default=['audiocaps', 'vggsound', 'laion']
    )
    parser.add_argument(
        '--cache_paths',
        nargs='+',
        type=str,
        default=None
    )

    parser.add_argument(
        "--t_cond", choices=['add', 'no', 'adaln'], default='add', help="How to condition on t")
    args = parser.parse_args()

    # Initialize DDP once per process
    rank, local_rank, world_size = setup_ddp()
    try:
        if rank == 0:
            print(
                f"DDP initialized. world_size={world_size}, node_local_rank={local_rank}, global_rank={rank}")
        main(rank, local_rank, world_size, args)
    finally:
        cleanup()
