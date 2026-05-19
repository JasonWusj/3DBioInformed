from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_MODALITIES = ["-t1n", "-t1c", "-t2w", "-t2f"]


def get_case_ids(data_dir):
    data_path = Path(data_dir)
    return sorted(d.name for d in data_path.iterdir() if d.is_dir() and d.name.startswith("BraTS"))


def split_cases(case_ids, train_ratio=0.7, val_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(case_ids))
    n_train = int(len(case_ids) * train_ratio)
    n_val = int(len(case_ids) * val_ratio)

    train_ids = [case_ids[i] for i in indices[:n_train]]
    val_ids = [case_ids[i] for i in indices[n_train:n_train + n_val]]
    test_ids = [case_ids[i] for i in indices[n_train + n_val:]]
    return train_ids, val_ids, test_ids


def find_nifti(case_path, suffix):
    case_path = Path(case_path)
    for path in case_path.iterdir():
        if path.name.endswith(".nii.gz") and suffix in path.name:
            return path
    raise FileNotFoundError(f"Missing NIfTI with suffix {suffix!r} in {case_path}")


def normalize_nonzero(image, clip=(-5.0, 5.0)):
    image = image.astype(np.float32, copy=True)
    for c in range(image.shape[0]):
        values = image[c]
        mask = values != 0
        if mask.any():
            mean = values[mask].mean()
            std = values[mask].std() + 1e-8
            values[mask] = (values[mask] - mean) / std
        image[c] = np.clip(values, clip[0], clip[1])
    return image


def load_case_arrays(case_path, modalities=None, seg_required=True):
    if modalities is None:
        modalities = DEFAULT_MODALITIES

    case_path = Path(case_path)
    images = []
    spacing_hwd = None
    for suffix in modalities:
        nii = nib.load(str(find_nifti(case_path, suffix)))
        if spacing_hwd is None:
            spacing_hwd = tuple(float(v) for v in nii.header.get_zooms()[:3])
        images.append(nii.get_fdata(dtype=np.float32))

    image = np.stack(images, axis=0)
    seg = None
    if seg_required:
        seg = nib.load(str(find_nifti(case_path, "-seg"))).get_fdata(dtype=np.float32).astype(np.uint8)
    return image, seg, spacing_hwd


def choose_crop_center(image, seg=None, mode="gt_tumor_center"):
    if mode == "none":
        return np.array(image.shape[-3:]) // 2

    if mode == "gt_tumor_center" and seg is not None:
        coords = np.argwhere(seg > 0)
        if len(coords) > 0:
            return np.rint(coords.mean(axis=0)).astype(int)

    foreground = np.any(image != 0, axis=0)
    coords = np.argwhere(foreground)
    if len(coords) > 0:
        return np.rint(coords.mean(axis=0)).astype(int)
    return np.array(image.shape[-3:]) // 2


def crop_or_pad_spatial(array, center_hwd, size_hwd, pad_value=0):
    spatial_shape = np.array(array.shape[-3:])
    center = np.asarray(center_hwd, dtype=int)
    size = np.asarray(size_hwd, dtype=int)
    start = center - size // 2
    end = start + size

    pad_before = np.maximum(-start, 0)
    pad_after = np.maximum(end - spatial_shape, 0)
    if np.any(pad_before > 0) or np.any(pad_after > 0):
        pad_width = [(0, 0)] * (array.ndim - 3)
        pad_width.extend((int(b), int(a)) for b, a in zip(pad_before, pad_after))
        array = np.pad(array, pad_width, mode="constant", constant_values=pad_value)
        start = start + pad_before

    slices = tuple(slice(int(s), int(s + n)) for s, n in zip(start, size))
    return array[(Ellipsis,) + slices]


def make_region_targets(seg):
    background = seg == 0
    tc = (seg == 1) | (seg == 4)
    wt = seg > 0
    et = seg == 4
    return np.stack([background, tc, wt, et], axis=0).astype(np.float32)


def to_cdhw(image_hwd):
    return np.transpose(image_hwd, (0, 3, 1, 2)).astype(np.float32)


def targets_to_cdhw(target_hwd):
    return np.transpose(target_hwd, (0, 3, 1, 2)).astype(np.float32)


def spacing_hwd_to_dhw(spacing_hwd):
    return (spacing_hwd[2], spacing_hwd[0], spacing_hwd[1])


def random_flip_3d(image, target):
    for axis in (1, 2, 3):
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=axis).copy()
            target = np.flip(target, axis=axis).copy()
    return image, target


class BraTS3DPatchDataset(Dataset):
    def __init__(
        self,
        data_dir,
        case_ids,
        patch_size=(128, 128, 128),
        modalities=None,
        crop_mode="gt_tumor_center",
        normalize_clip=(-5.0, 5.0),
        augment=False,
    ):
        self.data_dir = Path(data_dir)
        self.case_ids = list(case_ids)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.size_hwd = (self.patch_size[1], self.patch_size[2], self.patch_size[0])
        self.modalities = modalities or DEFAULT_MODALITIES
        self.crop_mode = crop_mode
        self.normalize_clip = tuple(normalize_clip)
        self.augment = augment

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        image, seg, _ = load_case_arrays(self.data_dir / case_id, self.modalities, seg_required=True)
        image = normalize_nonzero(image, self.normalize_clip)

        center = choose_crop_center(image, seg, self.crop_mode)
        image = crop_or_pad_spatial(image, center, self.size_hwd, pad_value=0)
        seg = crop_or_pad_spatial(seg, center, self.size_hwd, pad_value=0)

        image = to_cdhw(image)
        target = targets_to_cdhw(make_region_targets(seg))

        if self.augment:
            image, target = random_flip_3d(image, target)

        return torch.from_numpy(image), torch.from_numpy(target)


def load_case_volume(
    data_dir,
    case_id,
    modalities=None,
    normalize_clip=(-5.0, 5.0),
    crop_mode="gt_tumor_center",
    patch_size=(128, 128, 128),
):
    image, seg, spacing_hwd = load_case_arrays(Path(data_dir) / case_id, modalities, seg_required=True)
    image = normalize_nonzero(image, normalize_clip)

    if crop_mode != "none":
        size_hwd = (patch_size[1], patch_size[2], patch_size[0])
        center = choose_crop_center(image, seg, crop_mode)
        image = crop_or_pad_spatial(image, center, size_hwd, pad_value=0)
        seg = crop_or_pad_spatial(seg, center, size_hwd, pad_value=0)

    image = to_cdhw(image)
    target = targets_to_cdhw(make_region_targets(seg))
    return {
        "case_id": case_id,
        "image": torch.from_numpy(image),
        "target": torch.from_numpy(target),
        "spacing": spacing_hwd_to_dhw(spacing_hwd),
    }
