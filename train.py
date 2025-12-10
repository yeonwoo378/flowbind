import torch
from tqdm.auto import tqdm
import wandb
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.transforms import ToPILImage
from utils.train_utils import *

TOPIL = ToPILImage()


def _is_dist():
    return dist.is_available() and dist.is_initialized()


def _reduce_sum(x: torch.Tensor):
    if _is_dist():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def _all_min(x: torch.Tensor):
    # returns min across ranks in-place
    if _is_dist():
        dist.all_reduce(x, op=dist.ReduceOp.MIN)
    return x


MOD_BRIEF = {
    'ti': 'text_image',
    'ia': 'image_audio',
    'ta': 'text_audio',
}


def train_one_epoch(
    rank,
    models,
    dataloader,
    optimizer,
    scheduler,
    device,
    args,
    epoch,
    global_step,
    log_img_captions=None,
):
    is_main = (rank == 0)
    pbar = tqdm(
        dataloader, desc=f"Train Epoch {epoch}", leave=False, disable=not is_main)

    logging_info = {}

    for batch in pbar:
        # accumulate a single loss across all groups in this dataloader batch
        total_loss_scalar = 0.0
        total_bs_scalar = 0
        group_losses = []
        did_work = torch.tensor(0, device=device, dtype=torch.int)

        # {'ti': {'text': tensor, 'image': tensor}, 'ta': {...}, ...}
        for k, data_dict in batch.items():
            if not data_dict:
                continue

            feats_dict, means_dict, logvars_dict = {}, {}, {}

            # --- encode features ---
            # {'text': tensor, 'image': tensor}
            for mod, data in data_dict.items():
                with torch.no_grad():
                    feats = data.to(device, non_blocking=True)

                feats_dict[mod] = feats

                means, logvars = models[mod].module.logvar(
                    feats).chunk(2, dim=1)

                vars = torch.ones_like(
                    logvars).detach() * 10.0  # fixed variance
                logvars = torch.log(vars)

                logvars = torch.log(vars)

                means_dict[mod], logvars_dict[mod] = means, logvars

            feats = [feats_dict[m] for m in sorted(feats_dict)]
            means = [means_dict[m] for m in sorted(means_dict)]
            logvars = [logvars_dict[m] for m in sorted(logvars_dict)]

            z_mean, z_var = get_z_star(
                feats=feats, means=means, logvars=logvars, device=device, dropout=0.5)

            z = z_mean + torch.randn_like(z_mean) * torch.sqrt(z_var)

            flow_loss_dict = {}
            t_zero_flow_loss_dict = {}
            for mod, feat in feats_dict.items():
                t = torch.rand(feat.size(0), 1, device=device)

                one_sample_prob = 0.3
                mask = (torch.rand(t.size(), device=device)
                        < (1. - one_sample_prob)).float()
                t = t * mask + (1 - mask)

                xt = t * feat + (1 - t) * z
                noise = torch.zeros_like(xt)

                xt = xt.detach()
                pred = models[mod].module(t, xt)
                v_mse = F.mse_loss(
                    pred, feat - z.detach(), reduction='none')
                v_mse = v_mse.mean(dim=1)  # B,C -> B

                flow_loss_dict[mod] = v_mse.mean()
                t_zero = torch.zeros_like(t)
                t_zero_flow_loss = F.mse_loss(models[mod].module(
                    t_zero, z), feat - z)
                t_zero_flow_loss_dict[mod] = t_zero_flow_loss

            group_loss = sum(flow_loss_dict.values(
            )) + sum(t_zero_flow_loss_dict.values())

            group_losses.append(group_loss)
            total_loss_scalar += float(group_loss.detach()) * z.size(0)
            total_bs_scalar += int(z.size(0))
            did_work.fill_(1)

        _all_min(did_work)
        if did_work.item() == 0:
            # skip this batch consistently on all ranks
            global_step += 1
            continue

        optimizer.zero_grad(set_to_none=True)
        loss_total = torch.stack(group_losses).sum()
        loss_total.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # reduce average loss across ranks for logging
        tl = torch.tensor([total_loss_scalar],
                          device=device, dtype=torch.float32)
        tb = torch.tensor([total_bs_scalar], device=device,
                          dtype=torch.float32)
        _reduce_sum(tl)
        _reduce_sum(tb)
        avg_loss = (tl / (tb + 1e-9)).item()

        if is_main:
            wandb.log({
                'train/total_loss': avg_loss,
                'train/epoch': epoch,
                'train/global_step': global_step,
                'train/lr': optimizer.param_groups[0]['lr'],
            }, step=global_step)

        global_step += 1

    return logging_info, global_step
