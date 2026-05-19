import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from monai.inferers import sliding_window_inference
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import get_case_ids, load_case_volume, split_cases
from src.model import BiophysicsSegModel3D, StandardSegModel3D


REGIONS = ("TC", "WT", "ET")


def nested_region_masks(logits, threshold=0.5):
    probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
    et = probs[3] >= threshold
    tc = (probs[1] >= threshold) | et
    wt = (probs[2] >= threshold) | tc
    return {"TC": tc, "WT": wt, "ET": et}


def target_region_masks(target):
    target_np = target.detach().cpu().numpy()
    return {
        "TC": target_np[1] > 0.5,
        "WT": target_np[2] > 0.5,
        "ET": target_np[3] > 0.5,
    }


def dice_score(pred, target):
    intersection = np.logical_and(pred, target).sum()
    denominator = pred.sum() + target.sum()
    if denominator == 0:
        return 1.0
    return float(2.0 * intersection / denominator)


def surface(mask):
    if not mask.any():
        return mask
    structure = generate_binary_structure(3, 1)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_xor(mask, eroded)


def hd95(pred, target, spacing):
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return np.nan

    pred_surface = surface(pred)
    target_surface = surface(target)
    if not pred_surface.any() or not target_surface.any():
        return np.nan

    dt_target = distance_transform_edt(~target_surface, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate([dt_target[pred_surface], dt_pred[target_surface]])
    return float(np.percentile(distances, 95))


def flip_axes_for_tta(enabled):
    if not enabled:
        return [()]
    axes = (2, 3, 4)
    combos = []
    for r in range(0, len(axes) + 1):
        combos.extend(itertools.combinations(axes, r))
    return combos


@torch.no_grad()
def predict_logits(model, image, cfg, device):
    infer_cfg = cfg["inference"]
    image = image.unsqueeze(0).to(device)
    roi_size = tuple(infer_cfg["roi_size"])
    sw_batch_size = infer_cfg.get("sw_batch_size", 1)
    overlap = infer_cfg.get("overlap", 0.5)

    def predictor(window):
        return model(window, return_density=False)

    logits_sum = None
    axes_list = flip_axes_for_tta(infer_cfg.get("tta_flips", True))
    for axes in axes_list:
        aug_image = torch.flip(image, dims=axes) if axes else image
        logits = sliding_window_inference(
            aug_image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=predictor,
            overlap=overlap,
        )
        if axes:
            logits = torch.flip(logits, dims=axes)
        logits_sum = logits if logits_sum is None else logits_sum + logits
    return logits_sum / len(axes_list)


def evaluate_case(model, cfg, case_id, device):
    data_cfg = cfg["data"]
    infer_cfg = cfg["inference"]
    case = load_case_volume(
        data_dir=data_cfg["data_dir"],
        case_id=case_id,
        modalities=data_cfg["modalities"],
        normalize_clip=data_cfg["normalize_clip"],
        crop_mode=infer_cfg.get("eval_crop_mode", data_cfg.get("crop_mode", "gt_tumor_center")),
        patch_size=infer_cfg["roi_size"],
    )
    logits = predict_logits(model, case["image"], cfg, device)
    pred_masks = nested_region_masks(logits, threshold=infer_cfg.get("threshold", 0.5))
    target_masks = target_region_masks(case["target"])

    row = {"case_id": case_id}
    for region in REGIONS:
        row[f"{region}_dice"] = dice_score(pred_masks[region], target_masks[region])
        row[f"{region}_hd95"] = hd95(pred_masks[region], target_masks[region], case["spacing"])
    return row


def summarise(rows):
    summary = {}
    for region in REGIONS:
        dice_values = np.array([row[f"{region}_dice"] for row in rows], dtype=np.float64)
        hd_values = np.array([row[f"{region}_hd95"] for row in rows], dtype=np.float64)
        summary[region] = {
            "dice_mean": float(np.nanmean(dice_values)),
            "dice_std": float(np.nanstd(dice_values)),
            "hd95_mean": float(np.nanmean(hd_values)),
            "hd95_std": float(np.nanstd(hd_values)),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/paper3d_unet.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(cfg["output_dir"]) / "final_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg["loss"].get("use_biophysics", True):
        model = BiophysicsSegModel3D(cfg).to(device)
    else:
        model = StandardSegModel3D(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    case_ids = get_case_ids(cfg["data"]["data_dir"])
    _, val_ids, test_ids = split_cases(
        case_ids,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        seed=cfg["seed"],
    )
    eval_ids = val_ids if args.split == "val" else test_ids

    rows = []
    for case_id in tqdm(eval_ids, desc=f"Evaluating {args.split}"):
        rows.append(evaluate_case(model, cfg, case_id, device))

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.split}_metrics.csv"
    fieldnames = ["case_id"]
    for region in REGIONS:
        fieldnames.extend([f"{region}_dice", f"{region}_hd95"])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarise(rows)
    print(f"Saved per-case metrics to {csv_path}")
    print("Region  Dice mean/std    HD95 mean/std (mm)")
    for region in REGIONS:
        item = summary[region]
        print(
            f"{region:<6} "
            f"{item['dice_mean'] * 100:.2f}/{item['dice_std'] * 100:.2f}    "
            f"{item['hd95_mean']:.2f}/{item['hd95_std']:.2f}"
        )


if __name__ == "__main__":
    main()
