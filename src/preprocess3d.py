import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_case_patch, find_nifti, get_case_ids, spacing_hwd_to_dhw


def main():
    parser = argparse.ArgumentParser(description="Precompute 3D BraTS patches for faster training.")
    parser.add_argument("--config", type=str, default="configs/paper3d_unet.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    output_dir = Path(args.output_dir or data_cfg["preprocessed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    case_ids = get_case_ids(data_cfg["data_dir"])
    records = {}

    for case_id in tqdm(case_ids, desc="Preprocessing 3D patches"):
        image_name = f"{case_id}_image.npy"
        target_name = f"{case_id}_target.npy"
        image_path = output_dir / image_name
        target_path = output_dir / target_name

        spacing = None
        if args.overwrite or not image_path.exists() or not target_path.exists():
            image, target, spacing = build_case_patch(
                data_dir=data_cfg["data_dir"],
                case_id=case_id,
                patch_size=data_cfg["patch_size"],
                modalities=data_cfg["modalities"],
                crop_mode=data_cfg["crop_mode"],
                normalize_clip=data_cfg["normalize_clip"],
            )
            np.save(image_path, image.astype(np.float32, copy=False))
            np.save(target_path, target.astype(np.uint8, copy=False))
        if spacing is None:
            case_path = Path(data_cfg["data_dir"]) / case_id
            nii = nib.load(str(find_nifti(case_path, data_cfg["modalities"][0])))
            spacing_hwd = tuple(float(v) for v in nii.header.get_zooms()[:3])
            spacing = spacing_hwd_to_dhw(spacing_hwd)
        records[case_id] = {
            "image": image_name,
            "target": target_name,
            "spacing": list(spacing),
        }

    np.save(
        output_dir / "metadata.npy",
        {
            "cases": records,
            "patch_size": data_cfg["patch_size"],
            "crop_mode": data_cfg["crop_mode"],
            "modalities": data_cfg["modalities"],
            "normalize_clip": data_cfg["normalize_clip"],
        },
        allow_pickle=True,
    )
    print(f"Saved {len(case_ids)} preprocessed 3D .npy patches to {output_dir}")


if __name__ == "__main__":
    main()
