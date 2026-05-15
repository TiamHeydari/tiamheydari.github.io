# transformations.py
# ============================================================
# Paired transformation datasets for representation evaluation
# ============================================================
#
# Purpose:
#   Build paired samples:
#
#       x' = T_p(x)
#
#   and return:
#
#       x, x', p
#
#   where p is the transformation parameter.
#
# This is useful for:
#   1. Invariance:
#        cos(F_theta(x), F_theta(x')) ≈ 1
#
#   2. Equivariance:
#        can we predict p from z and z'?
#        can we predict z' from z and p?
#
# Supported transformations:
#   - dSprites factor interventions:
#       T_x_position, T_y_position, T_scale, T_orientation
#
#   - image corruptions:
#       T_noise, T_mask
#
# Notes:
#   dSprites latent class columns:
#       0: color
#       1: shape
#       2: scale
#       3: orientation
#       4: x-position
#       5: y-position
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset


# ------------------------------------------------------------
# dSprites factor metadata
# ------------------------------------------------------------

DSPRITES_FACTOR_COLS: Dict[str, int] = {
    "color": 0,
    "shape": 1,
    "scale": 2,
    "orientation": 3,
    "x_position": 4,
    "y_position": 5,
}


# ------------------------------------------------------------
# Pair record
# ------------------------------------------------------------

@dataclass
class PairRecord:
    """
    One paired transformation example.

    For factor interventions:
        idx_prime is the index of the pre-rendered transformed image.

    For corruptions:
        idx_prime is None, and x_prime is generated on the fly
        using transform_params.
    """

    idx: int
    idx_prime: Optional[int]

    transform_name: str
    transform_kind: str  # "factor" or "corruption"

    p: List[float]
    p_names: List[str]

    transform_params: Dict[str, Any]


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def build_latent_index(latents_classes: np.ndarray) -> Dict[Tuple[int, ...], int]:
    """
    Build dictionary:

        latent-class tuple -> image index

    This lets us find the exact dSprites image after changing
    one latent factor.
    """
    latent_to_index = {}

    for i, row in enumerate(latents_classes):
        key = tuple(row.astype(int).tolist())
        latent_to_index[key] = i

    return latent_to_index


def normalize_value(value: float, min_value: float, max_value: float) -> float:
    """
    Normalize a scalar to [0, 1].
    """
    if max_value == min_value:
        return 0.0

    return float((value - min_value) / (max_value - min_value))


def normalize_delta(delta: float, min_value: float, max_value: float) -> float:
    """
    Normalize a factor change by the valid factor range.

    Example:
        x-position has values 0,...,31.
        A shift of +8 gives 8 / 31.
    """
    if max_value == min_value:
        return 0.0

    return float(delta / (max_value - min_value))


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    """
    Convert one dSprites image to torch tensor [1, H, W].
    """
    x = torch.tensor(img, dtype=torch.float32)

    if x.ndim == 2:
        x = x.unsqueeze(0)

    return x


def image_to_hw_numpy(img: np.ndarray) -> np.ndarray:
    """
    Convert one image to numpy array [H, W].

    Accepts:
        [H, W]
        [1, H, W]
    """
    img = np.asarray(img)

    if img.ndim == 2:
        return img

    if img.ndim == 3 and img.shape[0] == 1:
        return img[0]

    raise ValueError(
        f"Expected one image with shape [H, W] or [1, H, W], got {img.shape}."
    )


def get_image_shape(imgs: np.ndarray) -> Tuple[int, int]:
    """
    Return image shape (H, W) from an image array.

    Accepts:
        [N, H, W]
        [N, 1, H, W]
    """
    imgs = np.asarray(imgs)

    if imgs.ndim == 3:
        return int(imgs.shape[1]), int(imgs.shape[2])

    if imgs.ndim == 4 and imgs.shape[1] == 1:
        return int(imgs.shape[2]), int(imgs.shape[3])

    raise ValueError(
        f"Expected imgs shape [N, H, W] or [N, 1, H, W], got {imgs.shape}."
    )


def get_object_center_from_image(
    img: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[int, int]:
    """
    Estimate object center from non-background pixels.

    For dSprites:
        object pixels are usually 1
        background pixels are 0

    Returns:
        cx, cy
    """
    img_hw = image_to_hw_numpy(img)

    ys, xs = np.where(img_hw > threshold)

    if len(xs) == 0:
        H, W = img_hw.shape
        return W // 2, H // 2

    cx = int(np.round(xs.mean()))
    cy = int(np.round(ys.mean()))

    return cx, cy


def compute_object_fraction_occluded(
    img: np.ndarray,
    x0: int,
    y0: int,
    patch_size: int,
    threshold: float = 0.5,
) -> float:
    """
    Compute what fraction of the object pixels are covered by the mask.

    This is useful for checking that a mask is not only covering background.
    """
    img_hw = image_to_hw_numpy(img)

    object_mask = img_hw > threshold
    object_area = int(object_mask.sum())

    if object_area == 0:
        return 0.0

    patch_mask = np.zeros_like(object_mask, dtype=bool)
    patch_mask[
        int(y0): int(y0) + int(patch_size),
        int(x0): int(x0) + int(patch_size),
    ] = True

    covered_object_area = int((object_mask & patch_mask).sum())
    fraction_occluded = covered_object_area / object_area

    return float(fraction_occluded)


# ------------------------------------------------------------
# Factor-pair construction
# ------------------------------------------------------------

def make_factor_pair_records(
    latents_classes: np.ndarray,
    factor_name: str,
    n_pairs: int = 2000,
    seed: int = 0,
    delta_choices: Optional[Sequence[int]] = None,
    target_value: Optional[int] = None,
    random_target: bool = False,
    p_mode: str = "delta_norm",
    max_tries_multiplier: int = 100,
) -> List[PairRecord]:
    """
    Create paired records by changing one dSprites factor.

    Example:
        x = G(shape, scale, orientation, x_position, y_position)

        x' = T_p(x)

    For x-position:
        x' = G(shape, scale, orientation, x_position + delta, y_position)

    Parameters
    ----------
    latents_classes:
        Array [N, 6] of dSprites latent classes.

    factor_name:
        One of:
            "x_position", "y_position", "scale", "orientation", "shape"

    n_pairs:
        Number of pairs to create.

    seed:
        Random seed.

    delta_choices:
        Candidate integer factor changes.

        Example:
            delta_choices=[-8, 8] for x-position.
            delta_choices=[-2, 2] for scale.

    target_value:
        Force transformed factor to a specific value.

    random_target:
        If True, randomly choose any different factor value.

    p_mode:
        How to represent p.

        Options:
            "delta_norm":
                p = normalized factor change

            "target_norm":
                p = normalized target factor value

            "from_to_delta_norm":
                p = [from_norm, to_norm, delta_norm]

    Returns
    -------
    records:
        List of PairRecord objects.
    """
    if factor_name not in DSPRITES_FACTOR_COLS:
        raise ValueError(
            f"Unknown factor_name={factor_name}. "
            f"Valid names are {list(DSPRITES_FACTOR_COLS.keys())}."
        )

    if sum(x is not None for x in [delta_choices, target_value]) > 1:
        raise ValueError(
            "Use only one of `delta_choices` or `target_value`."
        )

    if delta_choices is None and target_value is None and not random_target:
        raise ValueError(
            "Specify `delta_choices`, `target_value`, or set `random_target=True`."
        )

    rng = np.random.default_rng(seed)

    factor_col = DSPRITES_FACTOR_COLS[factor_name]
    latent_to_index = build_latent_index(latents_classes)

    unique_values = np.sort(np.unique(latents_classes[:, factor_col]).astype(int))
    min_value = int(unique_values.min())
    max_value = int(unique_values.max())

    records: List[PairRecord] = []

    n_total = len(latents_classes)
    max_tries = n_pairs * max_tries_multiplier
    tries = 0

    while len(records) < n_pairs and tries < max_tries:
        tries += 1

        idx = int(rng.integers(0, n_total))
        original_key = latents_classes[idx].astype(int).copy()

        old_value = int(original_key[factor_col])

        # -----------------------------
        # Choose transformed factor value
        # -----------------------------
        if delta_choices is not None:
            valid_new_values = []
            valid_deltas = []

            for delta in delta_choices:
                new_value = old_value + int(delta)

                if new_value in unique_values:
                    valid_new_values.append(new_value)
                    valid_deltas.append(int(delta))

            if len(valid_new_values) == 0:
                continue

            choice_idx = int(rng.integers(0, len(valid_new_values)))
            new_value = int(valid_new_values[choice_idx])
            delta = int(valid_deltas[choice_idx])

        elif target_value is not None:
            new_value = int(target_value)

            if new_value == old_value:
                continue

            if new_value not in unique_values:
                continue

            delta = int(new_value - old_value)

        else:
            possible_values = unique_values[unique_values != old_value]

            if len(possible_values) == 0:
                continue

            new_value = int(rng.choice(possible_values))
            delta = int(new_value - old_value)

        transformed_key = original_key.copy()
        transformed_key[factor_col] = new_value
        transformed_key_tuple = tuple(transformed_key.tolist())

        if transformed_key_tuple not in latent_to_index:
            continue

        idx_prime = int(latent_to_index[transformed_key_tuple])

        # -----------------------------
        # Define p
        # -----------------------------
        from_norm = normalize_value(old_value, min_value, max_value)
        to_norm = normalize_value(new_value, min_value, max_value)
        delta_norm = normalize_delta(delta, min_value, max_value)

        if p_mode == "delta_norm":
            p = [delta_norm]
            p_names = [f"delta_{factor_name}_norm"]

        elif p_mode == "target_norm":
            p = [to_norm]
            p_names = [f"target_{factor_name}_norm"]

        elif p_mode == "from_to_delta_norm":
            p = [from_norm, to_norm, delta_norm]
            p_names = [
                f"from_{factor_name}_norm",
                f"to_{factor_name}_norm",
                f"delta_{factor_name}_norm",
            ]

        else:
            raise ValueError(
                "Unknown p_mode. Use one of: "
                "'delta_norm', 'target_norm', 'from_to_delta_norm'."
            )

        transform_name = f"T_{factor_name}"

        records.append(
            PairRecord(
                idx=idx,
                idx_prime=idx_prime,
                transform_name=transform_name,
                transform_kind="factor",
                p=p,
                p_names=p_names,
                transform_params={
                    "factor_name": factor_name,
                    "factor_col": factor_col,
                    "from_value": old_value,
                    "to_value": new_value,
                    "delta": delta,
                    "from_norm": from_norm,
                    "to_norm": to_norm,
                    "delta_norm": delta_norm,
                },
            )
        )

    if len(records) < n_pairs:
        print(
            f"Warning: created only {len(records)} pairs out of requested {n_pairs} "
            f"for factor {factor_name}."
        )

    return records


# ------------------------------------------------------------
# Noise-pair construction
# ------------------------------------------------------------

def make_noise_pair_records(
    n_images: int,
    n_pairs: int = 2000,
    sigma: float = 0.15,
    seed: int = 0,
) -> List[PairRecord]:
    """
    Create paired records for Gaussian noise:

        x' = clip(x + epsilon, 0, 1)

        epsilon ~ N(0, sigma^2)

    Here p = [sigma].
    """
    rng = np.random.default_rng(seed)

    if n_pairs > n_images:
        indices = rng.choice(n_images, size=n_pairs, replace=True)
    else:
        indices = rng.choice(n_images, size=n_pairs, replace=False)

    records: List[PairRecord] = []

    for local_i, idx in enumerate(indices):
        noise_seed = int(seed + 10_000 + local_i)

        records.append(
            PairRecord(
                idx=int(idx),
                idx_prime=None,
                transform_name="T_noise",
                transform_kind="corruption",
                p=[float(sigma)],
                p_names=["sigma"],
                transform_params={
                    "sigma": float(sigma),
                    "noise_seed": noise_seed,
                },
            )
        )

    return records


# ------------------------------------------------------------
# Mask-pair construction
# ------------------------------------------------------------
def make_mask_pair_records(
    imgs: np.ndarray,
    n_pairs: int = 2000,
    patch_size: int = 14,
    seed: int = 0,
    mask_mode: str = "object_center",
    jitter: int = 6,
    threshold: float = 0.5,
    min_object_fraction_occluded: float = 0.05,
    max_center_tries: int = 50,
    transform_name: str = "T_mask",

    # New Gaussian options
    gaussian_center: bool = True,
    center_sigma: float = 4.0,
    gaussian_size: bool = True,
    size_sigma: float = 3.0,
    min_patch_size: int = 8,
    max_patch_size: int = 24,
) -> List[PairRecord]:
    """
    Create paired records for cutout / partial occlusion:

        x' = M ⊙ x

    Default behavior:
        mask_mode="object_center"

    The mask is placed near the object center.

    Optional Gaussian variability:
        center:
            dx, dy ~ Normal(0, center_sigma)

        size:
            patch_size_i ~ Normal(patch_size, size_sigma)

    The sampled patch size and position are stored in both:
        record.p
        record.transform_params

    p definition:
        p = [
            x0_norm,
            y0_norm,
            dx_from_object_center_norm,
            dy_from_object_center_norm,
            patch_size_norm
        ]
    """
    if mask_mode not in {"object_center", "random"}:
        raise ValueError(
            "mask_mode must be one of: 'object_center', 'random'."
        )

    rng = np.random.default_rng(seed)

    H, W = get_image_shape(imgs)
    n_images = len(imgs)

    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")

    if min_patch_size <= 0:
        raise ValueError("min_patch_size must be positive.")

    if max_patch_size < min_patch_size:
        raise ValueError("max_patch_size must be >= min_patch_size.")

    max_patch_size = int(min(max_patch_size, H, W))
    min_patch_size = int(min(min_patch_size, max_patch_size))

    if jitter < 0:
        raise ValueError("jitter must be non-negative.")

    if n_pairs > n_images:
        indices = rng.choice(n_images, size=n_pairs, replace=True)
    else:
        indices = rng.choice(n_images, size=n_pairs, replace=False)

    records: List[PairRecord] = []

    for idx in indices:
        idx = int(idx)
        img_hw = image_to_hw_numpy(imgs[idx])

        object_cx, object_cy = get_object_center_from_image(
            img_hw,
            threshold=threshold,
        )

        # ----------------------------------------------------
        # Sample patch size
        # ----------------------------------------------------
        if gaussian_size:
            sampled_patch_size = int(
                np.round(rng.normal(loc=patch_size, scale=size_sigma))
            )
        else:
            sampled_patch_size = int(patch_size)

        sampled_patch_size = int(
            np.clip(sampled_patch_size, min_patch_size, max_patch_size)
        )

        # Safety: make sure sampled patch fits inside image
        sampled_patch_size = int(min(sampled_patch_size, H, W))

        # ----------------------------------------------------
        # Random image-space mask
        # ----------------------------------------------------
        if mask_mode == "random":
            x0 = int(rng.integers(0, W - sampled_patch_size + 1))
            y0 = int(rng.integers(0, H - sampled_patch_size + 1))

            mask_cx = int(x0 + sampled_patch_size // 2)
            mask_cy = int(y0 + sampled_patch_size // 2)

            dx = int(mask_cx - object_cx)
            dy = int(mask_cy - object_cy)

            object_fraction_occluded = compute_object_fraction_occluded(
                img=img_hw,
                x0=x0,
                y0=y0,
                patch_size=sampled_patch_size,
                threshold=threshold,
            )

        # ----------------------------------------------------
        # Object-aware mask
        # ----------------------------------------------------
        else:
            best_candidate = None
            best_fraction = -1.0

            for _ in range(max_center_tries):
                if gaussian_center:
                    dx_try = int(np.round(rng.normal(loc=0.0, scale=center_sigma)))
                    dy_try = int(np.round(rng.normal(loc=0.0, scale=center_sigma)))
                else:
                    dx_try = int(rng.integers(-jitter, jitter + 1))
                    dy_try = int(rng.integers(-jitter, jitter + 1))

                mask_cx_try = int(object_cx + dx_try)
                mask_cy_try = int(object_cy + dy_try)

                x0_try = int(mask_cx_try - sampled_patch_size // 2)
                y0_try = int(mask_cy_try - sampled_patch_size // 2)

                x0_try = int(np.clip(x0_try, 0, W - sampled_patch_size))
                y0_try = int(np.clip(y0_try, 0, H - sampled_patch_size))

                mask_cx_try = int(x0_try + sampled_patch_size // 2)
                mask_cy_try = int(y0_try + sampled_patch_size // 2)

                dx_try = int(mask_cx_try - object_cx)
                dy_try = int(mask_cy_try - object_cy)

                frac_try = compute_object_fraction_occluded(
                    img=img_hw,
                    x0=x0_try,
                    y0=y0_try,
                    patch_size=sampled_patch_size,
                    threshold=threshold,
                )

                candidate = {
                    "x0": x0_try,
                    "y0": y0_try,
                    "mask_cx": mask_cx_try,
                    "mask_cy": mask_cy_try,
                    "dx": dx_try,
                    "dy": dy_try,
                    "object_fraction_occluded": frac_try,
                }

                if frac_try > best_fraction:
                    best_fraction = frac_try
                    best_candidate = candidate

                if frac_try >= min_object_fraction_occluded:
                    best_candidate = candidate
                    break

            if best_candidate is None:
                x0 = int(
                    np.clip(
                        object_cx - sampled_patch_size // 2,
                        0,
                        W - sampled_patch_size,
                    )
                )
                y0 = int(
                    np.clip(
                        object_cy - sampled_patch_size // 2,
                        0,
                        H - sampled_patch_size,
                    )
                )

                mask_cx = int(x0 + sampled_patch_size // 2)
                mask_cy = int(y0 + sampled_patch_size // 2)

                dx = int(mask_cx - object_cx)
                dy = int(mask_cy - object_cy)

                object_fraction_occluded = compute_object_fraction_occluded(
                    img=img_hw,
                    x0=x0,
                    y0=y0,
                    patch_size=sampled_patch_size,
                    threshold=threshold,
                )

            else:
                x0 = best_candidate["x0"]
                y0 = best_candidate["y0"]
                mask_cx = best_candidate["mask_cx"]
                mask_cy = best_candidate["mask_cy"]
                dx = best_candidate["dx"]
                dy = best_candidate["dy"]
                object_fraction_occluded = best_candidate["object_fraction_occluded"]

        # ----------------------------------------------------
        # Define p
        # ----------------------------------------------------
        x0_norm = normalize_value(x0, 0, W - 1)
        y0_norm = normalize_value(y0, 0, H - 1)

        dx_norm = normalize_delta(dx, 0, W - 1)
        dy_norm = normalize_delta(dy, 0, H - 1)

        patch_size_norm = float(sampled_patch_size / max(H, W))

        p = [
            x0_norm,
            y0_norm,
            dx_norm,
            dy_norm,
            patch_size_norm,
        ]

        p_names = [
            "x0_norm",
            "y0_norm",
            "dx_from_object_center_norm",
            "dy_from_object_center_norm",
            "patch_size_norm",
        ]

        records.append(
            PairRecord(
                idx=idx,
                idx_prime=None,
                transform_name=transform_name,
                transform_kind="corruption",
                p=p,
                p_names=p_names,
                transform_params={
                    "mask_mode": mask_mode,
                    "x0": int(x0),
                    "y0": int(y0),
                    "mask_cx": int(mask_cx),
                    "mask_cy": int(mask_cy),
                    "object_cx": int(object_cx),
                    "object_cy": int(object_cy),
                    "dx_from_object_center": int(dx),
                    "dy_from_object_center": int(dy),
                    "patch_size": int(sampled_patch_size),
                    "base_patch_size": int(patch_size),
                    "x0_norm": float(x0_norm),
                    "y0_norm": float(y0_norm),
                    "dx_from_object_center_norm": float(dx_norm),
                    "dy_from_object_center_norm": float(dy_norm),
                    "patch_size_norm": float(patch_size_norm),
                    "object_fraction_occluded": float(object_fraction_occluded),
                    "threshold": float(threshold),
                    "gaussian_center": bool(gaussian_center),
                    "center_sigma": float(center_sigma),
                    "gaussian_size": bool(gaussian_size),
                    "size_sigma": float(size_sigma),
                    "min_patch_size": int(min_patch_size),
                    "max_patch_size": int(max_patch_size),
                },
            )
        )

    return records

# ------------------------------------------------------------
# Apply corruption transforms
# ------------------------------------------------------------

def apply_noise_to_image(
    x: torch.Tensor,
    sigma: float,
    noise_seed: int,
) -> torch.Tensor:
    """
    Apply deterministic Gaussian noise to one image tensor [1, H, W].
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(noise_seed))

    noise = torch.randn(
        size=x.shape,
        generator=generator,
        dtype=x.dtype,
        device=x.device,
    )

    x_prime = x + float(sigma) * noise
    x_prime = torch.clamp(x_prime, 0.0, 1.0)

    return x_prime


def apply_mask_to_image(
    x: torch.Tensor,
    x0: int,
    y0: int,
    patch_size: int,
) -> torch.Tensor:
    """
    Apply cutout to one image tensor [1, H, W].
    """
    x_prime = x.clone()

    x_prime[
        :,
        int(y0): int(y0) + int(patch_size),
        int(x0): int(x0) + int(patch_size),
    ] = 0.0

    return x_prime


# ------------------------------------------------------------
# Paired transformation dataset
# ------------------------------------------------------------

class PairedTransformDataset(Dataset):
    """
    Dataset returning paired samples:

        x, x_prime, p

    Output dictionary:
        sample["x"]           : tensor [1, H, W]
        sample["x_prime"]     : tensor [1, H, W]
        sample["p"]           : tensor [p_dim]
        sample["idx"]         : original image index
        sample["idx_prime"]   : transformed image index, or -1 for corruptions
        sample["transform_id"]: integer ID of the transformation
    """

    def __init__(
        self,
        imgs: np.ndarray,
        records: Sequence[PairRecord],
    ):
        self.imgs = imgs
        self.records = list(records)

        if len(self.records) == 0:
            raise ValueError("PairedTransformDataset received zero records.")

        transform_names = sorted({r.transform_name for r in self.records})
        self.transform_name_to_id = {
            name: i for i, name in enumerate(transform_names)
        }
        self.id_to_transform_name = {
            i: name for name, i in self.transform_name_to_id.items()
        }

        self.p_names = self.records[0].p_names
        self.p_dim = len(self.records[0].p)

        for r in self.records:
            if len(r.p) != self.p_dim:
                raise ValueError(
                    "All records in one PairedTransformDataset must have the same p dimension. "
                    "If you mix transformations with different p dimensions, keep them as "
                    "separate datasets or write a custom collate function."
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]

        x = image_to_tensor(self.imgs[record.idx])

        if record.idx_prime is not None:
            x_prime = image_to_tensor(self.imgs[record.idx_prime])

        else:
            if record.transform_name.startswith("T_noise"):
                x_prime = apply_noise_to_image(
                    x=x,
                    sigma=record.transform_params["sigma"],
                    noise_seed=record.transform_params["noise_seed"],
                )

            elif record.transform_name.startswith("T_mask"):
                x_prime = apply_mask_to_image(
                    x=x,
                    x0=record.transform_params["x0"],
                    y0=record.transform_params["y0"],
                    patch_size=record.transform_params["patch_size"],
                )

            else:
                raise ValueError(
                    f"Unknown corruption transform: {record.transform_name}"
                )

        transform_id = self.transform_name_to_id[record.transform_name]

        sample = {
            "x": x,
            "x_prime": x_prime,
            "p": torch.tensor(record.p, dtype=torch.float32),
            "idx": torch.tensor(record.idx, dtype=torch.long),
            "idx_prime": torch.tensor(
                -1 if record.idx_prime is None else record.idx_prime,
                dtype=torch.long,
            ),
            "transform_id": torch.tensor(transform_id, dtype=torch.long),
        }

        return sample

    def get_record(self, index: int) -> PairRecord:
        """
        Return the original metadata record.
        """
        return self.records[index]

    def records_as_dataframe(self):
        """
        Convert record metadata to a pandas DataFrame.

        pandas is imported inside this function so the dataset itself
        does not require pandas unless this method is used.
        """
        import pandas as pd

        rows = []

        for r in self.records:
            row = {
                "idx": r.idx,
                "idx_prime": -1 if r.idx_prime is None else r.idx_prime,
                "transform_name": r.transform_name,
                "transform_kind": r.transform_kind,
            }

            for name, value in zip(r.p_names, r.p):
                row[name] = value

            for key, value in r.transform_params.items():
                row[key] = value

            rows.append(row)

        return pd.DataFrame(rows)


# ------------------------------------------------------------
# Factory functions
# ------------------------------------------------------------

def create_factor_pair_dataset(
    imgs: np.ndarray,
    latents_classes: np.ndarray,
    factor_name: str,
    n_pairs: int = 2000,
    seed: int = 0,
    delta_choices: Optional[Sequence[int]] = None,
    target_value: Optional[int] = None,
    random_target: bool = False,
    p_mode: str = "delta_norm",
) -> PairedTransformDataset:
    """
    Create a paired dataset for one dSprites factor intervention.

    Example:
        ds_x = create_factor_pair_dataset(
            imgs=imgs,
            latents_classes=latents_classes_all,
            factor_name="x_position",
            delta_choices=[-8, 8],
        )
    """
    records = make_factor_pair_records(
        latents_classes=latents_classes,
        factor_name=factor_name,
        n_pairs=n_pairs,
        seed=seed,
        delta_choices=delta_choices,
        target_value=target_value,
        random_target=random_target,
        p_mode=p_mode,
    )

    return PairedTransformDataset(imgs=imgs, records=records)


def create_noise_pair_dataset(
    imgs: np.ndarray,
    n_pairs: int = 2000,
    sigma: float = 0.15,
    seed: int = 0,
) -> PairedTransformDataset:
    """
    Create a paired dataset for Gaussian noise.
    """
    records = make_noise_pair_records(
        n_images=len(imgs),
        n_pairs=n_pairs,
        sigma=sigma,
        seed=seed,
    )

    return PairedTransformDataset(imgs=imgs, records=records)

def create_mask_pair_dataset(
    imgs: np.ndarray,
    n_pairs: int = 2000,
    patch_size: int = 14,
    seed: int = 0,
    mask_mode: str = "object_center",
    jitter: int = 6,
    threshold: float = 0.5,
    min_object_fraction_occluded: float = 0.05,
    max_center_tries: int = 50,
    transform_name: str = "T_mask",

    # New Gaussian options
    gaussian_center: bool = True,
    center_sigma: float = 4.0,
    gaussian_size: bool = True,
    size_sigma: float = 3.0,
    min_patch_size: int = 8,
    max_patch_size: int = 24,
) -> PairedTransformDataset:
    """
    Create a paired dataset for partial occlusion / cutout.

    Default:
        mask_mode="object_center"

    This places the mask near the shape center instead of randomly over
    the mostly empty background.

    Optional:
        Gaussian variability in mask center and mask size.
    """
    records = make_mask_pair_records(
        imgs=imgs,
        n_pairs=n_pairs,
        patch_size=patch_size,
        seed=seed,
        mask_mode=mask_mode,
        jitter=jitter,
        threshold=threshold,
        min_object_fraction_occluded=min_object_fraction_occluded,
        max_center_tries=max_center_tries,
        transform_name=transform_name,
        gaussian_center=gaussian_center,
        center_sigma=center_sigma,
        gaussian_size=gaussian_size,
        size_sigma=size_sigma,
        min_patch_size=min_patch_size,
        max_patch_size=max_patch_size,
    )

    return PairedTransformDataset(imgs=imgs, records=records)

def create_default_invariance_datasets(
    imgs: np.ndarray,
    latents_classes: np.ndarray,
    n_pairs: int = 2000,
    seed: int = 0,
    include_random_mask: bool = False,
) -> Dict[str, PairedTransformDataset]:
    """
    Create the default datasets for the invariance notebook.

    Recommended set:
        T_x_position : relevant dSprites factor
        T_scale      : relevant dSprites factor
        T_noise      : corruption stability
        T_mask       : object-aware partial occlusion

    Optional:
        T_mask_random: random image-space mask, mostly a background-control
    """
    datasets = {
        "T_x_position": create_factor_pair_dataset(
            imgs=imgs,
            latents_classes=latents_classes,
            factor_name="x_position",
            n_pairs=n_pairs,
            seed=seed,
            delta_choices=[-8, 8],
            p_mode="delta_norm",
        ),

        "T_scale": create_factor_pair_dataset(
            imgs=imgs,
            latents_classes=latents_classes,
            factor_name="scale",
            n_pairs=n_pairs,
            seed=seed + 1,
            delta_choices=[-2, 2],
            p_mode="delta_norm",
        ),

        "T_noise": create_noise_pair_dataset(
            imgs=imgs,
            n_pairs=n_pairs,
            sigma=0.15,
            seed=seed + 2,
        ),


        
        "T_mask": create_mask_pair_dataset(
            imgs=imgs,
            n_pairs=n_pairs,
            patch_size=14,
            seed=seed + 3,
            mask_mode="object_center",
            gaussian_center=True,
            center_sigma=4.0,
            gaussian_size=True,
            size_sigma=3.0,
            min_patch_size=8,
            max_patch_size=24,
            threshold=0.5,
            min_object_fraction_occluded=0.05,
            transform_name="T_mask",
        ),
    }

    if include_random_mask:
        datasets["T_mask_random"] = create_mask_pair_dataset(
            imgs=imgs,
            n_pairs=n_pairs,
            patch_size=14,
            seed=seed + 4,
            mask_mode="random",
            jitter=0,
            threshold=0.5,
            min_object_fraction_occluded=0.0,
            transform_name="T_mask_random",
        )

    return datasets


# ------------------------------------------------------------
# Quick visual check
# ------------------------------------------------------------

def show_pair_examples(
    pair_dataset: PairedTransformDataset,
    n_examples: int = 6,
    title: Optional[str] = None,
):
    """
    Show original and transformed images.

    Top row:
        x

    Bottom row:
        x' = T_p(x)
    """
    import matplotlib.pyplot as plt

    n_examples = min(n_examples, len(pair_dataset))

    fig, axes = plt.subplots(2, n_examples, figsize=(1.7 * n_examples, 3.2))

    if n_examples == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for i in range(n_examples):
        sample = pair_dataset[i]

        x = sample["x"][0].numpy()
        x_prime = sample["x_prime"][0].numpy()

        axes[0, i].imshow(x, cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")

        axes[1, i].imshow(x_prime, cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")

        if i == 0:
            axes[0, i].set_ylabel("x", fontsize=12)
            axes[1, i].set_ylabel("T(x)", fontsize=12)

    if title is None:
        title = pair_dataset.records[0].transform_name

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def summarize_mask_dataset(pair_dataset: PairedTransformDataset):
    """
    Print a small summary for mask datasets.

    Useful for checking that object-aware masks are actually occluding
    part of the dSprites object.
    """
    df = pair_dataset.records_as_dataframe()

    if "object_fraction_occluded" not in df.columns:
        print("This dataset does not contain mask occlusion metadata.")
        return df

    print("Mask dataset summary:")
    print("  n pairs:", len(df))
    print("  mask mode:", df["mask_mode"].iloc[0])
    print("  mean object fraction occluded:", df["object_fraction_occluded"].mean())
    print("  min object fraction occluded: ", df["object_fraction_occluded"].min())
    print("  max object fraction occluded: ", df["object_fraction_occluded"].max())

    return df


# ------------------------------------------------------------
# Minimal usage example
# ------------------------------------------------------------
#
# from transformations import (
#     create_default_invariance_datasets,
#     create_mask_pair_dataset,
#     show_pair_examples,
#     summarize_mask_dataset,
# )
#
# pair_datasets = create_default_invariance_datasets(
#     imgs=imgs,
#     latents_classes=latents_classes_all,
#     n_pairs=2000,
#     seed=7,
# )
#
# show_pair_examples(pair_datasets["T_mask"])
# mask_df = summarize_mask_dataset(pair_datasets["T_mask"])
#
# # Optional random-mask control:
# ds_random_mask = create_mask_pair_dataset(
#     imgs=imgs,
#     n_pairs=2000,
#     patch_size=14,
#     seed=11,
#     mask_mode="random",
#     transform_name="T_mask_random",
# )
#
# show_pair_examples(ds_random_mask)
# random_mask_df = summarize_mask_dataset(ds_random_mask)