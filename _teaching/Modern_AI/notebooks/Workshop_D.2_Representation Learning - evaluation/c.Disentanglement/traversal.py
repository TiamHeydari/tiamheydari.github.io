# ============================================================
# traversal.py
# Latent traversal utilities for c-Disentanglement notebook
#
# A. Coordinate directions
# B. PCA directions
# C. Factor-supervised directions
# ============================================================

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# Basic utilities
# ============================================================

def get_device() -> torch.device:
    """
    Use GPU if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def factor_vector_to_dict(
    y: torch.Tensor,
    factor_names: Sequence[str],
) -> Dict[str, int]:
    """
    Convert one factor vector y into a readable dictionary.

    y: [1, num_factors] or [num_factors]
    """
    if y.ndim == 2:
        y = y[0]

    return {
        str(name): int(y[i].item())
        for i, name in enumerate(factor_names)
    }


def get_sample_from_loader(
    loader,
    sample_index: int = 0,
    device: Optional[torch.device] = None,
):
    """
    Get one sample from a DataLoader.

    Returns:
        x: [1, 1, 64, 64]
        y: [1, 6]
        y_values: [1, 6] if available, otherwise None
    """
    if device is None:
        device = get_device()

    seen = 0

    for batch in loader:
        batch_size = batch["x"].shape[0]

        if seen + batch_size > sample_index:
            local_idx = sample_index - seen

            x = batch["x"][local_idx:local_idx + 1].to(device)
            y = batch["y"][local_idx:local_idx + 1]

            y_values = batch.get("y_values", None)
            if y_values is not None:
                y_values = y_values[local_idx:local_idx + 1]

            return x, y, y_values

        seen += batch_size

    raise IndexError("sample_index is larger than the dataset.")


def get_sample_by_factor(
    loader,
    factor_names: Sequence[str],
    factor_name: str,
    factor_value: int,
    device: Optional[torch.device] = None,
):
    """
    Get the first sample with a chosen factor value.

    Example:
        get_sample_by_factor(
            loader=test_loader,
            factor_names=dsprites["factor_names"],
            factor_name="scale",
            factor_value=0,
        )
    """
    if device is None:
        device = get_device()

    factor_names = [str(f) for f in factor_names]
    factor_idx = factor_names.index(factor_name)

    for batch in loader:
        y = batch["y"]
        mask = y[:, factor_idx] == factor_value

        if mask.any():
            idx = torch.where(mask)[0][0].item()

            x = batch["x"][idx:idx + 1].to(device)
            y_out = batch["y"][idx:idx + 1]

            y_values = batch.get("y_values", None)
            if y_values is not None:
                y_values = y_values[idx:idx + 1]

            return x, y_out, y_values

    raise ValueError(f"No sample found with {factor_name} = {factor_value}")


@torch.no_grad()
def encode_image(
    model: nn.Module,
    x: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Encode image into latent representation.

    Assumes:
        z = model.encode(x)
    """
    if device is None:
        device = get_device()

    model.to(device)
    model.eval()

    x = x.to(device)
    z = model.encode(x)

    return z


@torch.no_grad()
def decode_latents(
    model: nn.Module,
    z: torch.Tensor,
    device: Optional[torch.device] = None,
    output_are_logits: bool = True,
) -> torch.Tensor:
    """
    Decode latent vectors into images.

    Assumes model has either:
        model.decode(z)
    or:
        model.decoder(z)

    Returns:
        x_hat: [B, 1, 64, 64], values in [0, 1]
    """
    if device is None:
        device = get_device()

    model.to(device)
    model.eval()

    z = z.to(device)

    if hasattr(model, "decode"):
        out = model.decode(z)
    elif hasattr(model, "decoder"):
        out = model.decoder(z)
    else:
        raise AttributeError(
            "Latent traversal requires a decoder. "
            "The model must have model.decode(z) or model.decoder(z)."
        )

    if isinstance(out, tuple):
        out = out[0]

    if output_are_logits:
        x_hat = torch.sigmoid(out)
    else:
        x_hat = out

    return x_hat.detach().cpu()


@torch.no_grad()
def traverse_direction(
    model: nn.Module,
    x_base: torch.Tensor,
    direction: torch.Tensor,
    t_values: torch.Tensor,
    device: Optional[torch.device] = None,
    output_are_logits: bool = True,
) -> torch.Tensor:
    """
    General latent traversal:

        z_i = f_theta(x_i)
        z'(t) = z_i + t u
        x_hat'(t) = D(z'(t))

    Args:
        direction: u, shape [D_z]
        t_values: traversal coefficients, shape [n_steps]

    Returns:
        decoded images [n_steps, 1, 64, 64]
    """
    if device is None:
        device = get_device()

    model.to(device)
    model.eval()

    x_base = x_base.to(device)
    direction = direction.to(device).float()
    t_values = t_values.to(device).float()

    z_base = encode_image(model, x_base, device=device)  # [1, D_z]

    z_traversal = z_base + t_values[:, None] * direction[None, :]

    images = decode_latents(
        model=model,
        z=z_traversal,
        device=device,
        output_are_logits=output_are_logits,
    )

    return images


# ============================================================
# Plotting utilities
# ============================================================

def plot_image_row_with_original(
    x_original: torch.Tensor,
    images: torch.Tensor,
    col_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
):
    """
    Plot:

        original | generated traversal outputs

    x_original: [1, 1, H, W]
    images: [n_steps, 1, H, W]
    """
    x_original = x_original.detach().cpu()
    images = images.detach().cpu()

    n_steps = images.shape[0]
    n_cols = n_steps + 1

    if figsize is None:
        figsize = (1.45 * n_cols, 1.8)

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(
        x_original[0].squeeze().numpy(),
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[0].axis("off")
    axes[0].set_title("original", fontsize=9)

    for j in range(n_steps):
        axes[j + 1].imshow(
            images[j].squeeze().numpy(),
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[j + 1].axis("off")

        if col_labels is not None:
            axes[j + 1].set_title(col_labels[j], fontsize=9)

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()

from typing import List, Optional, Tuple
import torch
import matplotlib.pyplot as plt


def plot_image_grid_with_original(
    x_original: torch.Tensor,
    image_rows: List[torch.Tensor],
    row_labels: List[str],
    col_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
):
    """
    Plot multiple traversal rows.

    Each row is one traversal direction:

        original | t1 | t2 | ... | tN

    Row labels are manually drawn with ax.text()
    because ax.axis("off") hides normal y-axis labels.
    """

    x_original = x_original.detach().cpu()

    num_rows = len(image_rows)
    num_steps = image_rows[0].shape[0]
    num_cols = num_steps + 1

    if figsize is None:
        figsize = (1.45 * num_cols, 1.45 * num_rows + 1.2)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=figsize,
        squeeze=False,
    )

    for r in range(num_rows):
        # -----------------------------
        # Original image
        # -----------------------------
        ax = axes[r, 0]

        ax.imshow(
            x_original[0].squeeze().numpy(),
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        ax.set_xticks([])
        ax.set_yticks([])

        # Do NOT use set_ylabel here.
        # axis("off") or tight_layout can hide/clip it.
        ax.text(
            -0.20,
            0.5,
            row_labels[r],
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

        if r == 0:
            ax.set_title("original", fontsize=9)

        # -----------------------------
        # Traversed/generated images
        # -----------------------------
        images = image_rows[r].detach().cpu()

        for c in range(num_steps):
            ax = axes[r, c + 1]

            ax.imshow(
                images[c].squeeze().numpy(),
                cmap="gray",
                vmin=0,
                vmax=1,
            )

            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0 and col_labels is not None:
                ax.set_title(col_labels[c], fontsize=9)

    if title is not None:
        fig.suptitle(title, fontsize=13)

    # Important: leave space on the left for row labels.
    plt.subplots_adjust(
        left=0.12,
        right=0.98,
        top=0.86 if title is not None else 0.95,
        bottom=0.05,
        wspace=0.05,
        hspace=0.10,
    )

    plt.show()
# ============================================================
# A. Coordinate directions
# ============================================================

def coordinate_direction(
    latent_dim: int,
    m: int,
) -> torch.Tensor:
    """
    Coordinate direction:

        u_m = e_m

    Returns:
        e_m: [latent_dim]
    """
    if m < 0 or m >= latent_dim:
        raise ValueError(f"m must be between 0 and {latent_dim - 1}")

    u = torch.zeros(latent_dim)
    u[m] = 1.0

    return u


@torch.no_grad()
def coordinate_traversal(
    model: nn.Module,
    x_base: torch.Tensor,
    Z_reference: torch.Tensor,
    dims: Sequence[int],
    n_steps: int = 9,
    alpha_min: float = -3.0,
    alpha_max: float = 3.0,
    device: Optional[torch.device] = None,
    output_are_logits: bool = True,
):
    """
    A. Coordinate traversal.

    For each coordinate direction:

        u_m = e_m

    we traverse:

        z'(t) = z + t e_m

    Here t is chosen in units of the empirical standard deviation
    of latent coordinate z_m in Z_reference:

        t = alpha * std(z_m)

    Returns:
        image_rows: list of decoded image rows
        t_rows: list of t-values used for each dimension
    """
    if device is None:
        device = get_device()

    Z_reference = Z_reference.detach().float().cpu()

    latent_dim = Z_reference.shape[1]

    image_rows = []
    t_rows = []

    alpha_values = torch.linspace(alpha_min, alpha_max, n_steps)

    for m in dims:
        u_m = coordinate_direction(latent_dim=latent_dim, m=m)

        std_m = Z_reference[:, m].std().clamp_min(1e-8)

        t_values = alpha_values * std_m

        images = traverse_direction(
            model=model,
            x_base=x_base,
            direction=u_m,
            t_values=t_values,
            device=device,
            output_are_logits=output_are_logits,
        )

        image_rows.append(images)
        t_rows.append(t_values)

    return image_rows, t_rows


def plot_coordinate_traversal(
    model: nn.Module,
    x_base: torch.Tensor,
    Z_reference: torch.Tensor,
    dims: Sequence[int],
    n_steps: int = 9,
    alpha_min: float = -3.0,
    alpha_max: float = 3.0,
    model_name: str = "Model",
    device: Optional[torch.device] = None,
    output_are_logits: bool = True,
):
    """
    Compute and plot coordinate traversal.

    A. Coordinate directions:
        u_m = e_m
    """
    image_rows, t_rows = coordinate_traversal(
        model=model,
        x_base=x_base,
        Z_reference=Z_reference,
        dims=dims,
        n_steps=n_steps,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        device=device,
        output_are_logits=output_are_logits,
    )

    row_labels = [f"z{m}" for m in dims]

    alpha_values = torch.linspace(alpha_min, alpha_max, n_steps)
    col_labels = [f"{a:.1f}σ" for a in alpha_values.tolist()]

    plot_image_grid_with_original(
        x_original=x_base,
        image_rows=image_rows,
        row_labels=row_labels,
        col_labels=col_labels,
        title=f"{model_name}: A. Coordinate traversal",
    )

    return image_rows, t_rows


# ============================================================
# B. PCA directions
# ============================================================

def compute_pca_directions(
    Z: torch.Tensor,
    n_components: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    B. PCA on latent codes.

    Given latent codes:

        Z = {z_i}

    compute PCA directions:

        u_m = PCA_m(Z)

    Returns:
        directions: [n_components, D_z]
        mean: [D_z]
        explained_variance: [n_components]
        explained_variance_ratio: [n_components]
        scores: [N, n_components]
    """
    Z_cpu = Z.detach().float().cpu()

    N, D_z = Z_cpu.shape

    if n_components is None:
        n_components = D_z

    n_components = min(n_components, D_z)

    mean = Z_cpu.mean(dim=0)
    Z_centered = Z_cpu - mean

    # SVD:
    # Z_centered = U S V^T
    # rows of V^T are PCA directions
    _, S, Vh = torch.linalg.svd(Z_centered, full_matrices=False)

    directions = Vh[:n_components]  # [n_components, D_z]

    explained_variance_all = (S ** 2) / max(N - 1, 1)
    explained_variance = explained_variance_all[:n_components]

    total_variance = explained_variance_all.sum().clamp_min(1e-12)
    explained_variance_ratio = explained_variance / total_variance

    scores = Z_centered @ directions.T

    return {
        "directions": directions,
        "mean": mean,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "scores": scores,
    }


@torch.no_grad()
def pca_traversal(
    model: nn.Module,
    x_base: torch.Tensor,
    Z_reference: torch.Tensor,
    pc_indices: Sequence[int],
    n_steps: int = 9,
    alpha_min: float = -3.0,
    alpha_max: float = 3.0,
    device: Optional[torch.device] = None,
    output_are_logits: bool = True,
):
    """
    B. PCA traversal.

    PCA directions:

        u_m = PCA_m(Z)

    Traversal:

        z'(t) = z + t u_m

    Here t is chosen in units of the empirical standard deviation
    of PCA scores along PC_m:

        t = alpha * std(score_m)
    """
    if device is None:
        device = get_device()

    max_pc = max(pc_indices)

    pca_info = compute_pca_directions(
        Z=Z_reference,
        n_components=max_pc + 1,
    )

    directions = pca_info["directions"]
    scores = pca_info["scores"]

    image_rows = []
    t_rows = []

    alpha_values = torch.linspace(alpha_min, alpha_max, n_steps)

    for pc_idx in pc_indices:
        u_m = directions[pc_idx]

        score_std = scores[:, pc_idx].std().clamp_min(1e-8)

        t_values = alpha_values * score_std

        images = traverse_direction(
            model=model,
            x_base=x_base,
            direction=u_m,
            t_values=t_values,
            device=device,
            output_are_logits=output_are_logits,
        )

        image_rows.append(images)
        t_rows.append(t_values)

    return image_rows, t_rows, pca_info


def plot_pca_traversal(
    model: nn.Module,
    x_base: torch.Tensor,
    Z_reference: torch.Tensor,
    pc_indices: Sequence[int] = (0, 1, 2),
    n_steps: int = 9,
    alpha_min: float = -3.0,
    alpha_max: float = 3.0,
    model_name: str = "Model",
    device: Optional[torch.device] = None,
    output_are_logits: bool = True,
):
    """
    Compute and plot PCA traversal.

    B. Unsupervised directions:
        u_m = PCA_m(Z)
    """
    image_rows, t_rows, pca_info = pca_traversal(
        model=model,
        x_base=x_base,
        Z_reference=Z_reference,
        pc_indices=pc_indices,
        n_steps=n_steps,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        device=device,
        output_are_logits=output_are_logits,
    )

    row_labels = [f"PC{pc_idx + 1}" for pc_idx in pc_indices]

    alpha_values = torch.linspace(alpha_min, alpha_max, n_steps)
    col_labels = [f"{a:.1f}σ" for a in alpha_values.tolist()]

    plot_image_grid_with_original(
        x_original=x_base,
        image_rows=image_rows,
        row_labels=row_labels,
        col_labels=col_labels,
        title=f"{model_name}: B. PCA traversal",
    )

    return image_rows, t_rows, pca_info


def plot_pca_explained_variance(
    pca_info: Dict[str, torch.Tensor],
    n_components: Optional[int] = None,
    title: str = "PCA explained variance",
):
    """
    Plot PCA explained variance ratio.
    """
    evr = pca_info["explained_variance_ratio"].detach().cpu()

    if n_components is not None:
        evr = evr[:n_components]

    x = torch.arange(1, len(evr) + 1)

    plt.figure(figsize=(5, 3))
    plt.plot(x.numpy(), evr.numpy(), marker="o")
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# C. Factor-supervised directions
# ============================================================

def compute_factor_means(
    Z: torch.Tensor,
    Y: torch.Tensor,
    factor_names: Sequence[str],
    factor_name: str,
) -> Dict[int, torch.Tensor]:
    """
    C. Factor-supervised latent means.

    First encode all samples:

        z_i = f_theta(x_i)

    For factor FV_k with values c = 0, ..., C_k - 1:

        mu_{k,c}
        =
        mean { z_i : FV_k(x_i) = c }

    Returns:
        means[c] = mu_{k,c}
    """
    factor_names = [str(f) for f in factor_names]
    factor_idx = factor_names.index(factor_name)

    Z_cpu = Z.detach().float().cpu()
    y_cpu = Y[:, factor_idx].detach().cpu().long()

    values = torch.unique(y_cpu).tolist()

    means = {}

    for c in values:
        mask = y_cpu == c
        means[int(c)] = Z_cpu[mask].mean(dim=0)

    return means


def get_factor_direction(
    Z: torch.Tensor,
    Y: torch.Tensor,
    factor_names: Sequence[str],
    factor_name: str,
    start_value: int,
    end_value: int,
    normalize: bool = False,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    C. Factor-supervised direction between two factor values.

    Compute:

        mu_{k,start} = mean { z_i : FV_k(x_i) = start_value }
        mu_{k,end}   = mean { z_i : FV_k(x_i) = end_value }

    Define direction:

        u_k = mu_{k,end} - mu_{k,start}

    Examples:
        u_scale = mu_scale,5 - mu_scale,0

        u_square_to_heart = mu_shape,2 - mu_shape,0

    normalize:
        False is recommended for t in [0, 1],
        because t = 1 moves by exactly the mean-difference vector.
    """
    means = compute_factor_means(
        Z=Z,
        Y=Y,
        factor_names=factor_names,
        factor_name=factor_name,
    )

    if start_value not in means:
        raise ValueError(
            f"start_value={start_value} not found for factor {factor_name}"
        )

    if end_value not in means:
        raise ValueError(
            f"end_value={end_value} not found for factor {factor_name}"
        )

    direction = means[end_value] - means[start_value]

    original_norm = direction.norm().item()

    if normalize:
        direction = direction / (direction.norm() + 1e-8)

    info = {
        "factor_name": factor_name,
        "start_value": start_value,
        "end_value": end_value,
        "direction_norm_before_normalization": original_norm,
        "normalize": normalize,
    }

    return direction, info


@torch.no_grad()
def factor_supervised_traversal(
    model: nn.Module,
    x_base: torch.Tensor,
    direction: torch.Tensor,
    t_values: Optional[torch.Tensor] = None,
    n_steps: int = 9,
    device: Optional[torch.device] = None,
    output_are_logits: bool = True,
) -> torch.Tensor:
    """
    C. Factor-supervised traversal.

    Given:

        u_k = mu_{k,end} - mu_{k,start}

    traverse:

        z'(t) = z + t u_k

    Default:
        t in [0, 1]
    """
    if device is None:
        device = get_device()

    if t_values is None:
        t_values = torch.linspace(0.0, 1.0, n_steps)

    images = traverse_direction(
        model=model,
        x_base=x_base,
        direction=direction,
        t_values=t_values,
        device=device,
        output_are_logits=output_are_logits,
    )

    return images


def plot_factor_supervised_traversal(
    model: nn.Module,
    x_base: torch.Tensor,
    direction: torch.Tensor,
    direction_name: str,
    t_values: Optional[torch.Tensor] = None,
    n_steps: int = 9,
    model_name: str = "Model",
    device: Optional[torch.device] = None,
    output_are_logits: bool = True,
):
    """
    Compute and plot factor-supervised traversal.

    C. Factor-supervised directions:
        u_k = mu_{k,end} - mu_{k,start}
    """
    if t_values is None:
        t_values = torch.linspace(0.0, 1.0, n_steps)

    images = factor_supervised_traversal(
        model=model,
        x_base=x_base,
        direction=direction,
        t_values=t_values,
        n_steps=n_steps,
        device=device,
        output_are_logits=output_are_logits,
    )

    col_labels = [f"t={t:.2f}" for t in t_values.tolist()]

    plot_image_row_with_original(
        x_original=x_base,
        images=images,
        col_labels=col_labels,
        title=f"{model_name}: C. {direction_name}",
    )

    return images