import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (
    BraTS3DPreprocessedPatchDataset,
    BraTS3DPatchDataset,
    get_case_ids,
    get_case_ids_from_preprocessed,
    split_cases,
)
from src.losses import (
    BiophysicsInformedLoss3D,
    DiceLoss,
    DiceWithBCELoss,
    FocalLoss,
    GaussianRefinementRegularization,
    JaccardLoss,
    StructureAwareDiceBCELoss,
)
from src.model import BiophysicsSegModel3D, StandardSegModel3D


def log(message):
    print(message, flush=True)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(cfg, model):
    training_cfg = cfg["training"]
    name = training_cfg.get("optimizer", "ranger").lower()
    if name == "ranger":
        try:
            from torch_optimizer import Ranger
        except ImportError as exc:
            raise ImportError("Ranger requires torch-optimizer. Install requirement.txt first.") from exc
        return Ranger(
            model.parameters(),
            lr=training_cfg["lr"],
            weight_decay=training_cfg.get("weight_decay", 0.0),
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=training_cfg["lr"],
            weight_decay=training_cfg.get("weight_decay", 0.0),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_datasets(cfg):
    data_cfg = cfg["data"]
    use_preprocessed = data_cfg.get("use_preprocessed", False)
    if use_preprocessed:
        case_ids = get_case_ids_from_preprocessed(data_cfg["preprocessed_dir"])
    else:
        case_ids = get_case_ids(data_cfg["data_dir"])
    train_ids, val_ids, test_ids = split_cases(
        case_ids,
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        seed=data_cfg.get("split_seed", cfg["seed"]),
    )
    train_fraction = float(data_cfg.get("train_fraction", 1.0))
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError(f"data.train_fraction must be in (0, 1], got {train_fraction}")
    if train_fraction < 1.0:
        n_train = max(1, int(len(train_ids) * train_fraction))
        train_ids = train_ids[:n_train]

    if use_preprocessed:
        common = {
            "preprocessed_dir": data_cfg["preprocessed_dir"],
            "mmap": data_cfg.get("mmap_preprocessed", True),
        }
        train_dataset = BraTS3DPreprocessedPatchDataset(case_ids=train_ids, augment=data_cfg.get("augment", True), **common)
        val_dataset = BraTS3DPreprocessedPatchDataset(case_ids=val_ids, augment=False, **common)
    else:
        common = {
            "data_dir": data_cfg["data_dir"],
            "patch_size": data_cfg["patch_size"],
            "modalities": data_cfg["modalities"],
            "crop_mode": data_cfg["crop_mode"],
            "normalize_clip": data_cfg["normalize_clip"],
        }
        train_dataset = BraTS3DPatchDataset(case_ids=train_ids, augment=data_cfg.get("augment", True), **common)
        val_dataset = BraTS3DPatchDataset(case_ids=val_ids, augment=False, **common)
    return train_dataset, val_dataset, train_ids, val_ids, test_ids


def build_segmentation_loss(loss_cfg):
    if isinstance(loss_cfg, str):
        name = loss_cfg.lower()
        loss_cfg = {}
    else:
        name = loss_cfg.get("segmentation_loss", "dice").lower()
    if name == "dice":
        return DiceLoss()
    if name == "dice_ce":
        return DiceWithBCELoss()
    if name == "structure_aware_dice_ce":
        return StructureAwareDiceBCELoss(
            lambda_hierarchy=loss_cfg.get("lambda_hierarchy", 0.2),
            lambda_boundary=loss_cfg.get("lambda_boundary", 0.5),
            boundary_width=loss_cfg.get("boundary_width", 1),
        )
    if name == "focal":
        return FocalLoss()
    if name == "jaccard":
        return JaccardLoss()
    raise ValueError(f"Unsupported segmentation loss: {name}")


def dataloader_kwargs(cfg, device, shuffle):
    data_cfg = cfg["data"]
    num_workers = int(data_cfg["num_workers"])
    kwargs = {
        "batch_size": cfg["training"]["batch_size"],
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda" and data_cfg.get("pin_memory", True),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = data_cfg.get("persistent_workers", True)
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    return kwargs


def dice_per_region(logits, target, threshold=0.5):
    probs = torch.sigmoid(logits)
    pred = probs >= threshold

    et = pred[:, 3]
    tc = pred[:, 1] | et
    wt = pred[:, 2] | tc
    pred_regions = [tc, wt, et]
    target_regions = [target[:, 1] > 0.5, target[:, 2] > 0.5, target[:, 3] > 0.5]

    scores = []
    for pred_mask, target_mask in zip(pred_regions, target_regions):
        intersection = (pred_mask & target_mask).sum(dim=(1, 2, 3)).float()
        denominator = pred_mask.sum(dim=(1, 2, 3)).float() + target_mask.sum(dim=(1, 2, 3)).float()
        score = torch.where(denominator > 0, 2.0 * intersection / denominator, torch.ones_like(denominator))
        scores.extend(score.detach().cpu().tolist())
    return scores


def train_one_epoch(model, loader, criterion, gaussian_regularization, optimizer, scaler, device, use_amp, cfg, epoch):
    model.train()
    losses = {"seg": 0.0, "pde": 0.0, "bc": 0.0, "gaussian": 0.0, "total": 0.0}
    num_batches = 0
    start = time.time()
    use_biophysics = cfg["loss"].get("use_biophysics", True)

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            if use_biophysics:
                outputs = model(images, return_density=True)
                gaussian_aux = None
                if len(outputs) == 4:
                    logits, u_hat, t_tensor, gaussian_aux = outputs
                else:
                    logits, u_hat, t_tensor = outputs
                loss, loss_dict = criterion(logits, targets, u_hat, t_tensor)
                if gaussian_regularization is not None and gaussian_aux is not None:
                    gaussian_loss = gaussian_regularization(gaussian_aux)
                    loss = loss + gaussian_loss
                    loss_dict["gaussian"] = float(gaussian_loss.detach().cpu())
                    loss_dict["total"] = float(loss.detach().cpu())
                else:
                    loss_dict["gaussian"] = 0.0
            else:
                logits = model(images, return_density=False)
                loss = criterion(logits, targets)
                loss_dict = {
                    "seg": float(loss.detach().cpu()),
                    "pde": 0.0,
                    "bc": 0.0,
                    "gaussian": 0.0,
                    "total": float(loss.detach().cpu()),
                }

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"].get("grad_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"].get("grad_clip", 1.0))
            optimizer.step()

        for key in losses:
            losses[key] += loss_dict[key]
        num_batches += 1

        interval = cfg["training"].get("log_interval", 10)
        if (batch_idx + 1) % interval == 0 or (batch_idx + 1) == len(loader):
            log(
                f"[Epoch {epoch + 1} Batch {batch_idx + 1}/{len(loader)}] "
                f"total={loss_dict['total']:.4f} seg={loss_dict['seg']:.4f} "
                f"pde={loss_dict['pde']:.6f} bc={loss_dict['bc']:.6f} "
                f"gaussian={loss_dict['gaussian']:.6f}"
            )

    for key in losses:
        losses[key] /= max(num_batches, 1)
    return losses, time.time() - start


@torch.no_grad()
def validate(model, loader, dice_loss, device, use_amp):
    model.eval()
    val_loss = 0.0
    scores = []
    num_batches = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images, return_density=False)
            loss = dice_loss(logits, targets)
        val_loss += float(loss.detach().cpu())
        scores.extend(dice_per_region(logits, targets))
        num_batches += 1

    return val_loss / max(num_batches, 1), float(np.mean(scores)) if scores else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/paper3d_unet.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["training"].get("amp", True) and device.type == "cuda")

    log("3D biophysics-informed segmentation training")
    log(f"Config: {args.config}")
    log(f"Device: {device}")

    train_dataset, val_dataset, train_ids, val_ids, test_ids = build_datasets(cfg)
    log(f"Cases: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    train_loader = DataLoader(
        train_dataset,
        **dataloader_kwargs(cfg, device, shuffle=True),
    )
    val_loader = DataLoader(
        val_dataset,
        **dataloader_kwargs(cfg, device, shuffle=False),
    )

    if cfg["loss"].get("use_biophysics", True):
        model = BiophysicsSegModel3D(cfg).to(device)
        log("Model: 3D UNet + density estimator")
    else:
        model = StandardSegModel3D(cfg).to(device)
        log("Model: standard 3D UNet")
    optimizer = build_optimizer(cfg, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    scaler = GradScaler(enabled=use_amp)

    loss_cfg = cfg["loss"]
    segmentation_loss = build_segmentation_loss(loss_cfg)
    gaussian_regularization = None
    if cfg["model"].get("gaussian_refinement", {}).get("enabled", False):
        gaussian_regularization = GaussianRefinementRegularization(
            lambda_sigma=loss_cfg.get("lambda_gaussian_sigma", 1.0e-4),
            lambda_amplitude=loss_cfg.get("lambda_gaussian_amplitude", 1.0e-3),
        ).to(device)
    if loss_cfg.get("use_biophysics", True):
        criterion = BiophysicsInformedLoss3D(
            lambda_pde=loss_cfg["lambda_pde"],
            lambda_bc=loss_cfg["lambda_bc"],
            d_range=loss_cfg["d_range"],
            rho_range=loss_cfg["rho_range"],
            sample_parameters=loss_cfg.get("sample_parameters", "voxel"),
            segmentation_loss=segmentation_loss,
        ).to(device)
    else:
        criterion = segmentation_loss.to(device)
    dice_loss = DiceLoss().to(device)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    csv_path = output_dir / "training_log.csv"
    best_dice = -1.0
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "train_total", "train_seg_loss", "train_pde", "train_bc", "train_gaussian", "val_dice_loss", "val_mean_dice", "lr", "epoch_time_s"])

        for epoch in range(cfg["training"]["epochs"]):
            train_losses, epoch_time = train_one_epoch(model, train_loader, criterion, gaussian_regularization, optimizer, scaler, device, use_amp, cfg, epoch)
            val_dice_loss, val_mean_dice = validate(model, val_loader, dice_loss, device, use_amp)
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            writer.writerow([
                epoch + 1,
                f"{train_losses['total']:.6f}",
                f"{train_losses['seg']:.6f}",
                f"{train_losses['pde']:.6f}",
                f"{train_losses['bc']:.6f}",
                f"{train_losses['gaussian']:.6f}",
                f"{val_dice_loss:.6f}",
                f"{val_mean_dice:.6f}",
                f"{lr:.8f}",
                f"{epoch_time:.2f}",
            ])
            csv_file.flush()

            log(
                f"[Epoch {epoch + 1}] train_total={train_losses['total']:.4f} "
                f"val_dice_loss={val_dice_loss:.4f} val_mean_dice={val_mean_dice:.4f} lr={lr:.2e}"
            )

            if val_mean_dice > best_dice:
                best_dice = val_mean_dice
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_dice": best_dice,
                        "config": cfg,
                    },
                    output_dir / "best_model.pth",
                )

            interval = cfg["training"].get("checkpoint_interval", 25)
            if (epoch + 1) % interval == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_dice": best_dice,
                        "config": cfg,
                    },
                    output_dir / f"checkpoint_epoch{epoch + 1}.pth",
                )

    torch.save(
        {
            "epoch": cfg["training"]["epochs"] - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
        },
        output_dir / "final_model.pth",
    )
    log(f"Training complete. Best validation mean Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
