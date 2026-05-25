"""
Evaluate trained model on val/test split.
Reports Dice (↑) and Hausdorff Distance 95 (mm, ↓) for TC / WT / ET.

Usage:
    python src/evaluate.py --config configs/paper3d_unet.yaml --checkpoint outputs/paper3d_unet/final_model.pth --split test
    python src/evaluate.py --config configs/baseline3d_unet.yaml --split test
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from torch.amp import autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (
    BraTS3DPreprocessedPatchDataset,
    find_nifti,
    get_case_ids,
    get_case_ids_from_preprocessed,
    load_case_volume,
    spacing_hwd_to_dhw,
    split_cases,
)
from src.model import BiophysicsSegModel3D, StandardSegModel3D

REGIONS = ("TC", "WT", "ET")


def nested_region_masks(logits, threshold=0.5):
    """Convert 4-channel logits to nested binary region masks."""
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    if probs.ndim == 5:
        probs = probs[0]
    et = probs[3] >= threshold
    tc = (probs[1] >= threshold) | et
    wt = (probs[2] >= threshold) | tc
    return {"TC": tc, "WT": wt, "ET": et}


def target_region_masks(target):
    """Convert 4-channel target to nested binary region masks."""
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if target.ndim == 5:
        target = target[0]
    return {
        "TC": target[1] > 0.5,
        "WT": target[2] > 0.5,
        "ET": target[3] > 0.5,
    }


def dice_score(pred, target):
    intersection = np.logical_and(pred, target).sum()
    denominator = pred.sum() + target.sum()
    if denominator == 0:
        return 1.0
    return float(2.0 * intersection / denominator)


def surface_voxels(mask):
    if not mask.any():
        return mask
    structure = generate_binary_structure(3, 1)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_xor(mask, eroded)


def hausdorff_distance_95(pred, target, spacing=(1.0, 1.0, 1.0)):
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return np.nan

    pred_surf = surface_voxels(pred)
    target_surf = surface_voxels(target)
    if not pred_surf.any() or not target_surf.any():
        return np.nan

    dt_target = distance_transform_edt(~target_surf, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_surf, sampling=spacing)
    distances = np.concatenate([dt_target[pred_surf], dt_pred[target_surf]])
    return float(np.percentile(distances, 95))


def load_model(cfg, checkpoint_path, device):
    if cfg["loss"].get("use_biophysics", True):
        model = BiophysicsSegModel3D(cfg).to(device)
    else:
        model = StandardSegModel3D(cfg).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    best_dice = checkpoint.get("best_dice", "?")
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Epoch: {epoch}, Best val dice: {best_dice}")
    return model


def spacing_from_raw_case(data_cfg, case_id):
    import nibabel as nib

    case_path = Path(data_cfg["data_dir"]) / case_id
    nii = nib.load(str(find_nifti(case_path, data_cfg["modalities"][0])))
    spacing_hwd = tuple(float(v) for v in nii.header.get_zooms()[:3])
    return spacing_hwd_to_dhw(spacing_hwd)


def preprocessed_spacing(data_cfg, dataset, case_id):
    record = dataset.records[case_id]
    if "spacing" in record:
        return tuple(float(v) for v in record["spacing"])
    if Path(data_cfg["data_dir"]).exists():
        return spacing_from_raw_case(data_cfg, case_id)
    raise RuntimeError(
        "HD95 in mm requires voxel spacing. Re-run src/preprocess3d.py with raw data "
        "available, or evaluate with --mode raw."
    )


@torch.no_grad()
def evaluate_preprocessed(model, cfg, eval_ids, device):
    """Evaluate on preprocessed patches (single forward pass per case)."""
    data_cfg = cfg["data"]
    infer_cfg = cfg["inference"]
    threshold = infer_cfg.get("threshold", 0.5)
    use_amp = cfg["training"].get("amp", True) and device.type == "cuda"

    dataset = BraTS3DPreprocessedPatchDataset(
        preprocessed_dir=data_cfg["preprocessed_dir"],
        case_ids=eval_ids,
        augment=False,
    )

    rows = []
    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        image, target = dataset[idx]
        image = image.unsqueeze(0).to(device)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(image, return_density=False)

        pred_masks = nested_region_masks(logits, threshold=threshold)
        tgt_masks = target_region_masks(target)

        row = {"case_id": eval_ids[idx]}
        spacing = preprocessed_spacing(data_cfg, dataset, eval_ids[idx])
        for region in REGIONS:
            row[f"{region}_dice"] = dice_score(pred_masks[region], tgt_masks[region])
            row[f"{region}_hd95"] = hausdorff_distance_95(
                pred_masks[region], tgt_masks[region], spacing=spacing
            )
        rows.append(row)
    return rows


@torch.no_grad()
def evaluate_raw(model, cfg, eval_ids, device):
    """Evaluate on raw NIfTI volumes with sliding window inference + TTA."""
    try:
        from monai.inferers import sliding_window_inference
    except ImportError:
        print("MONAI not installed. Falling back to single-patch evaluation.")
        print("Install with: pip install monai")
        return evaluate_preprocessed(model, cfg, eval_ids, device)

    import itertools

    data_cfg = cfg["data"]
    infer_cfg = cfg["inference"]
    threshold = infer_cfg.get("threshold", 0.5)
    roi_size = tuple(infer_cfg["roi_size"])
    sw_batch_size = infer_cfg.get("sw_batch_size", 1)
    overlap = infer_cfg.get("overlap", 0.5)
    tta_flips = infer_cfg.get("tta_flips", True)

    def get_flip_axes(enabled):
        if not enabled:
            return [()]
        axes = (2, 3, 4)
        combos = []
        for r in range(len(axes) + 1):
            combos.extend(itertools.combinations(axes, r))
        return combos

    axes_list = get_flip_axes(tta_flips)

    def predictor(window):
        return model(window, return_density=False)

    rows = []
    for case_id in tqdm(eval_ids, desc="Evaluating (sliding window + TTA)"):
        case = load_case_volume(
            data_dir=data_cfg["data_dir"],
            case_id=case_id,
            modalities=data_cfg["modalities"],
            normalize_clip=data_cfg["normalize_clip"],
            crop_mode=infer_cfg.get("eval_crop_mode", "none"),
            patch_size=infer_cfg["roi_size"],
        )
        image = case["image"].unsqueeze(0).to(device)

        logits_sum = None
        for axes in axes_list:
            aug_image = torch.flip(image, dims=axes) if axes else image
            logits = sliding_window_inference(
                aug_image, roi_size=roi_size, sw_batch_size=sw_batch_size,
                predictor=predictor, overlap=overlap,
            )
            if axes:
                logits = torch.flip(logits, dims=axes)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        logits_avg = logits_sum / len(axes_list)

        pred_masks = nested_region_masks(logits_avg, threshold=threshold)
        tgt_masks = target_region_masks(case["target"])

        row = {"case_id": case_id}
        for region in REGIONS:
            row[f"{region}_dice"] = dice_score(pred_masks[region], tgt_masks[region])
            row[f"{region}_hd95"] = hausdorff_distance_95(
                pred_masks[region], tgt_masks[region], spacing=case["spacing"]
            )
        rows.append(row)
    return rows


def summarise(rows):
    summary = {}
    for region in REGIONS:
        dice_vals = np.array([r[f"{region}_dice"] for r in rows], dtype=np.float64)
        hd_vals = np.array([r[f"{region}_hd95"] for r in rows], dtype=np.float64)
        summary[region] = {
            "dice_mean": float(np.nanmean(dice_vals)),
            "dice_std": float(np.nanstd(dice_vals)),
            "hd95_mean": float(np.nanmean(hd_vals)),
            "hd95_std": float(np.nanstd(hd_vals)),
            "n_valid_hd": int(np.sum(~np.isnan(hd_vals))),
        }
    return summary


def print_results(summary, split_name):
    print(f"\n{'='*60}")
    print(f"  Evaluation Results ({split_name})")
    print(f"{'='*60}")
    print(f"{'Region':<8} {'Dice (%) ↑':<20} {'HD95 (mm) ↓':<20}")
    print(f"{'-'*60}")
    for region in REGIONS:
        s = summary[region]
        dice_str = f"{s['dice_mean']*100:.2f} ± {s['dice_std']*100:.2f}"
        hd_str = f"{s['hd95_mean']:.2f} ± {s['hd95_std']:.2f}"
        if np.isnan(s['hd95_mean']):
            hd_str = "N/A (empty predictions)"
        print(f"{region:<8} {dice_str:<20} {hd_str:<20}")
    print(f"{'-'*60}")

    mean_dice = np.mean([summary[r]["dice_mean"] for r in REGIONS])
    valid_hd = [summary[r]["hd95_mean"] for r in REGIONS if not np.isnan(summary[r]["hd95_mean"])]
    mean_hd = np.mean(valid_hd) if valid_hd else float("nan")
    print(f"{'Mean':<8} {mean_dice*100:.2f}{'':16} {mean_hd:.2f}")
    print(f"{'='*60}\n")


def save_csv(rows, csv_path):
    fieldnames = ["case_id"]
    for region in REGIONS:
        fieldnames.extend([f"{region}_dice", f"{region}_hd95"])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-case metrics saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model (Dice ↑, HD95 ↓)")
    parser.add_argument("--config", type=str, default="configs/paper3d_unet.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint. Default: <output_dir>/final_model.pth")
    parser.add_argument("--split", choices=["val", "test", "both"], default="test")
    parser.add_argument("--mode", choices=["preprocessed", "raw", "auto"], default="auto",
                        help="'preprocessed': fast eval on patches. 'raw': sliding window + TTA on NIfTI.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        output_dir = Path(cfg["output_dir"])
        final = output_dir / "final_model.pth"
        checkpoint_path = final
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(cfg, checkpoint_path, device)

    # Resolve case IDs
    data_cfg = cfg["data"]
    use_preprocessed = data_cfg.get("use_preprocessed", False)
    if use_preprocessed and Path(data_cfg["preprocessed_dir"]).exists():
        case_ids = get_case_ids_from_preprocessed(data_cfg["preprocessed_dir"])
    else:
        case_ids = get_case_ids(data_cfg["data_dir"])

    _, val_ids, test_ids = split_cases(
        case_ids,
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        seed=data_cfg.get("split_seed", cfg["seed"]),
    )

    # Determine evaluation mode
    if args.mode == "auto":
        raw_available = Path(data_cfg["data_dir"]).exists()
        mode = "raw" if raw_available else "preprocessed"
    else:
        mode = args.mode
    print(f"Evaluation mode: {mode}")

    # Run evaluation
    splits_to_eval = ["val", "test"] if args.split == "both" else [args.split]
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in splits_to_eval:
        eval_ids = val_ids if split == "val" else test_ids
        print(f"\nEvaluating {split} split ({len(eval_ids)} cases)...")

        if mode == "raw":
            rows = evaluate_raw(model, cfg, eval_ids, device)
        else:
            rows = evaluate_preprocessed(model, cfg, eval_ids, device)

        summary = summarise(rows)
        print_results(summary, split)
        save_csv(rows, output_dir / f"{split}_metrics.csv")


if __name__ == "__main__":
    main()
