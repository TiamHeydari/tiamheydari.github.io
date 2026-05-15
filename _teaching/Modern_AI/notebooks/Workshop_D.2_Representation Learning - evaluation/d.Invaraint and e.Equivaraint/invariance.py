# invariance.py
# ============================================================
# Invariance analysis for learned representations
# ============================================================
#
# Purpose:
#   Given paired transformation datasets:
#
#       x_prime = T_p(x)
#
#   and a frozen representation model:
#
#       z       = F_theta(x)
#       z_prime = F_theta(x_prime)
#
#   compute:
#
#       Inv(T) = mean_i cos(z_i, z'_i)
#
# Interpretation:
#   Inv(T) close to 1:
#       representation is stable under transformation T
#
#   Inv(T) much lower than 1:
#       representation is sensitive to transformation T
#
# Notes:
#   For VAE-style models, F_theta(x) should be the deterministic
#   latent mean mu_phi(x), not a sampled latent vector.
# ============================================================

from __future__ import annotations

from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Device helper
# ------------------------------------------------------------

def get_default_device() -> torch.device:
    """
    Return CUDA device if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# Representation extraction
# ------------------------------------------------------------

def _extract_latent_from_encoder_output(output: Any) -> torch.Tensor:
    """
    Convert common encoder outputs into a latent representation tensor.

    Supported cases:
        1. Tensor:
            z

        2. Tuple/list:
            (mu, logvar)
            or (z, ...)
            We use the first element.

        3. Dict:
            uses one of the common keys:
                "z", "mu", "mean", "latent"
    """
    if torch.is_tensor(output):
        z = output

    elif isinstance(output, (tuple, list)):
        if len(output) == 0:
            raise ValueError("Encoder returned an empty tuple/list.")
        z = output[0]

    elif isinstance(output, dict):
        possible_keys = ["z", "mu", "mean", "latent"]

        found_key = None
        for key in possible_keys:
            if key in output:
                found_key = key
                break

        if found_key is None:
            raise ValueError(
                "Encoder returned a dict, but none of the expected keys "
                "were found: 'z', 'mu', 'mean', 'latent'."
            )

        z = output[found_key]

    else:
        raise TypeError(
            f"Unsupported encoder output type: {type(output)}"
        )

    if not torch.is_tensor(z):
        raise TypeError(
            f"Extracted latent representation is not a tensor: {type(z)}"
        )

    # If encoder returns spatial features, flatten them.
    if z.ndim > 2:
        z = z.flatten(start_dim=1)

    return z


@torch.no_grad()
def encode_F_theta(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute deterministic representation:

        z = F_theta(x)

    For AE:
        usually encoder output.

    For β-VAE / β-TCVAE:
        this should be mu_phi(x), usually returned as the first
        output of model.encode(x).

    Parameters
    ----------
    model:
        Frozen representation model.

    x:
        Image batch [B, 1, H, W].

    device:
        Device for evaluation.

    Returns
    -------
    z:
        Representation tensor [B, d].
    """
    if device is None:
        device = get_default_device()

    model = model.to(device)
    model.eval()

    x = x.to(device)

    if hasattr(model, "encode"):
        output = model.encode(x)

    elif hasattr(model, "encoder"):
        output = model.encoder(x)

    else:
        raise AttributeError(
            "Model has neither `.encode(x)` nor `.encoder(x)`. "
            "Add one of these methods so the representation can be extracted."
        )

    z = _extract_latent_from_encoder_output(output)

    return z


@torch.no_grad()
def encode_loader_pairs(
    model: torch.nn.Module,
    pair_loader,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Encode all pairs in a paired transformation DataLoader.

    Each batch must contain:
        batch["x"]
        batch["x_prime"]
        batch["p"]

    Returns
    -------
    encoded:
        {
            "Z":       [N, d],
            "Z_prime": [N, d],
            "P":       [N, p_dim],
        }
    """
    if device is None:
        device = get_default_device()

    Z_list: List[torch.Tensor] = []
    Z_prime_list: List[torch.Tensor] = []
    P_list: List[torch.Tensor] = []

    model = model.to(device)
    model.eval()

    for batch in pair_loader:
        x = batch["x"]
        x_prime = batch["x_prime"]
        p = batch["p"]

        z = encode_F_theta(model, x, device=device)
        z_prime = encode_F_theta(model, x_prime, device=device)

        Z_list.append(z.detach().cpu())
        Z_prime_list.append(z_prime.detach().cpu())
        P_list.append(p.detach().cpu())

    encoded = {
        "Z": torch.cat(Z_list, dim=0),
        "Z_prime": torch.cat(Z_prime_list, dim=0),
        "P": torch.cat(P_list, dim=0),
    }

    return encoded


# ------------------------------------------------------------
# Invariance metrics
# ------------------------------------------------------------

def compute_cosine_invariance_from_Z(
    Z: torch.Tensor,
    Z_prime: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """
    Compute cosine-based invariance score from encoded pairs.

    Parameters
    ----------
    Z:
        Original representations [N, d].

    Z_prime:
        Transformed representations [N, d].

    eps:
        Numerical stability constant.

    Returns
    -------
    metrics:
        Dictionary containing mean/std/min/max cosine similarity
        and all per-sample similarities.
    """
    if Z.shape != Z_prime.shape:
        raise ValueError(
            f"Z and Z_prime must have the same shape. "
            f"Got {Z.shape} and {Z_prime.shape}."
        )

    similarities = F.cosine_similarity(Z, Z_prime, dim=1, eps=eps)

    metrics = {
        "mean_cosine": float(similarities.mean().item()),
        "std_cosine": float(similarities.std(unbiased=False).item()),
        "min_cosine": float(similarities.min().item()),
        "max_cosine": float(similarities.max().item()),
        "median_cosine": float(similarities.median().item()),
        "n_pairs": int(similarities.shape[0]),
        "similarities": similarities.detach().cpu().numpy(),
    }

    return metrics


def compute_l2_shift_from_Z(
    Z: torch.Tensor,
    Z_prime: torch.Tensor,
) -> Dict[str, Any]:
    """
    Compute L2 representation shift:

        ||z - z_prime||_2

    This is optional. Cosine similarity is the main invariance metric.
    """
    if Z.shape != Z_prime.shape:
        raise ValueError(
            f"Z and Z_prime must have the same shape. "
            f"Got {Z.shape} and {Z_prime.shape}."
        )

    shifts = torch.linalg.norm(Z - Z_prime, dim=1)

    metrics = {
        "mean_l2_shift": float(shifts.mean().item()),
        "std_l2_shift": float(shifts.std(unbiased=False).item()),
        "min_l2_shift": float(shifts.min().item()),
        "max_l2_shift": float(shifts.max().item()),
        "median_l2_shift": float(shifts.median().item()),
        "l2_shifts": shifts.detach().cpu().numpy(),
    }

    return metrics


@torch.no_grad()
def compute_invariance_for_loader(
    model: torch.nn.Module,
    pair_loader,
    device: Optional[torch.device] = None,
    return_encoded: bool = False,
) -> Dict[str, Any]:
    """
    Compute invariance metrics for one model and one transformation loader.

    Parameters
    ----------
    model:
        Frozen representation model.

    pair_loader:
        DataLoader over a PairedTransformDataset.

    device:
        Evaluation device.

    return_encoded:
        If True, also return Z, Z_prime, and P.
        Useful later for equivariance.

    Returns
    -------
    result:
        Dictionary with cosine invariance metrics.
    """
    encoded = encode_loader_pairs(
        model=model,
        pair_loader=pair_loader,
        device=device,
    )

    Z = encoded["Z"]
    Z_prime = encoded["Z_prime"]

    cosine_metrics = compute_cosine_invariance_from_Z(Z, Z_prime)
    l2_metrics = compute_l2_shift_from_Z(Z, Z_prime)

    result = {
        **cosine_metrics,
        **l2_metrics,
    }

    if return_encoded:
        result["Z"] = Z
        result["Z_prime"] = Z_prime
        result["P"] = encoded["P"]

    return result


# ------------------------------------------------------------
# Run full invariance analysis
# ------------------------------------------------------------

def run_invariance_analysis(
    models: Dict[str, torch.nn.Module],
    pair_loaders: Dict[str, Any],
    device: Optional[torch.device] = None,
    return_encoded: bool = False,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Dict[str, Any]]]]:
    """
    Run invariance analysis for multiple models and transformations.

    Parameters
    ----------
    models:
        Dictionary:
            model_name -> model

    pair_loaders:
        Dictionary:
            transformation_name -> DataLoader

    device:
        Evaluation device.

    return_encoded:
        If True, store Z, Z_prime, P for each model/transformation.

    verbose:
        Print progress.

    Returns
    -------
    invariance_df:
        Summary table.

    invariance_results:
        Nested dictionary:
            invariance_results[model_name][transformation_name]
    """
    if device is None:
        device = get_default_device()

    records = []
    invariance_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for model_name, model in models.items():
        if verbose:
            print(f"\nEvaluating model: {model_name}")

        invariance_results[model_name] = {}

        for transform_name, pair_loader in pair_loaders.items():
            result = compute_invariance_for_loader(
                model=model,
                pair_loader=pair_loader,
                device=device,
                return_encoded=return_encoded,
            )

            invariance_results[model_name][transform_name] = result

            record = {
                "model": model_name,
                "transformation": transform_name,
                "mean_cosine": result["mean_cosine"],
                "std_cosine": result["std_cosine"],
                "median_cosine": result["median_cosine"],
                "min_cosine": result["min_cosine"],
                "max_cosine": result["max_cosine"],
                "mean_l2_shift": result["mean_l2_shift"],
                "std_l2_shift": result["std_l2_shift"],
                "median_l2_shift": result["median_l2_shift"],
                "n_pairs": result["n_pairs"],
            }

            records.append(record)

            if verbose:
                print(
                    f"  {transform_name:15s} | "
                    f"Inv = {result['mean_cosine']:.4f} ± {result['std_cosine']:.4f} | "
                    f"L2 shift = {result['mean_l2_shift']:.4f}"
                )

    invariance_df = pd.DataFrame(records)

    return invariance_df, invariance_results


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_invariance_bar(
    invariance_df: pd.DataFrame,
    value_col: str = "mean_cosine",
    title: str = "Intrinsic invariance score",
    ylim: Tuple[float, float] = (-1.0, 1.05),
    figsize: Tuple[float, float] = (9, 4),
):
    """
    Bar plot of invariance scores.

    Rows:
        transformations

    Columns:
        models

    Values:
        mean cosine similarity
    """
    pivot = invariance_df.pivot(
        index="transformation",
        columns="model",
        values=value_col,
    )

    ax = pivot.plot(
        kind="bar",
        figsize=figsize,
        ylim=ylim,
    )

    ax.set_title(title)
    ax.set_xlabel("Transformation T")
    ax.set_ylabel(value_col)

    if value_col == "mean_cosine":
        ax.axhline(1.0, linestyle="--", linewidth=1)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

    return pivot


def plot_l2_shift_bar(
    invariance_df: pd.DataFrame,
    title: str = "Representation shift under transformation",
    figsize: Tuple[float, float] = (9, 4),
):
    """
    Bar plot of mean L2 shift.

    Lower means more stable in Euclidean distance.
    """
    pivot = invariance_df.pivot(
        index="transformation",
        columns="model",
        values="mean_l2_shift",
    )

    ax = pivot.plot(
        kind="bar",
        figsize=figsize,
    )

    ax.set_title(title)
    ax.set_xlabel("Transformation T")
    ax.set_ylabel(r"mean $\|z - z'\|_2$")

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

    return pivot


def plot_similarity_distributions(
    invariance_results: Dict[str, Dict[str, Dict[str, Any]]],
    bins: int = 40,
    figsize: Tuple[float, float] = (8, 4),
):
    """
    Plot cosine similarity distributions for each model.

    One figure per model.
    """
    for model_name, transform_results in invariance_results.items():
        plt.figure(figsize=figsize)

        for transform_name, result in transform_results.items():
            sims = result["similarities"]

            plt.hist(
                sims,
                bins=bins,
                alpha=0.5,
                density=True,
                label=transform_name,
            )

        plt.title(f"Cosine similarity distributions: {model_name}")
        plt.xlabel(r"$\cos(F_\theta(x), F_\theta(T(x)))$")
        plt.ylabel("density")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_invariance_heatmap(
    invariance_df: pd.DataFrame,
    value_col: str = "mean_cosine",
    title: str = "Intrinsic invariance heatmap",
    figsize: Tuple[float, float] = (6, 4),
):
    """
    Simple heatmap of invariance scores.

    Uses matplotlib only.
    """
    pivot = invariance_df.pivot(
        index="transformation",
        columns="model",
        values=value_col,
    )

    values = pivot.values

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(values, aspect="auto", vmin=-1.0, vmax=1.0)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))

    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)

    plt.setp(
        ax.get_xticklabels(),
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j,
                i,
                f"{values[i, j]:.2f}",
                ha="center",
                va="center",
            )

    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=value_col)

    plt.tight_layout()
    plt.show()

    return pivot


# ------------------------------------------------------------
# Interpretation helper
# ------------------------------------------------------------

def print_invariance_interpretation():
    """
    Print a short interpretation guide.
    """
    print("Interpretation:")
    print("  Inv(T) = mean cosine similarity between F_theta(x) and F_theta(T(x)).")
    print()
    print("  Inv(T) close to 1:")
    print("      representation is stable under transformation T.")
    print()
    print("  Inv(T) much lower than 1:")
    print("      representation changes when T is applied.")
    print()
    print("Important:")
    print("  Low invariance is not automatically bad.")
    print("  For reconstruction-trained models, sensitivity to position or scale")
    print("  may be expected because the decoder needs those factors.")
    print()
    print("  Disentanglement and invariance are different:")
    print("      disentanglement = factor is separated")
    print("      invariance      = factor change does not move z")


# ------------------------------------------------------------
# Convenience function
# ------------------------------------------------------------

def run_and_plot_invariance(
    models: Dict[str, torch.nn.Module],
    pair_loaders: Dict[str, Any],
    device: Optional[torch.device] = None,
    return_encoded: bool = False,
    verbose: bool = True,
):
    """
    Convenience wrapper:
        1. run invariance analysis
        2. plot bar summary
        3. plot heatmap
        4. print interpretation

    Returns
    -------
    invariance_df, invariance_results
    """
    invariance_df, invariance_results = run_invariance_analysis(
        models=models,
        pair_loaders=pair_loaders,
        device=device,
        return_encoded=return_encoded,
        verbose=verbose,
    )

    plot_invariance_bar(invariance_df)
    plot_invariance_heatmap(invariance_df)
    print_invariance_interpretation()

    return invariance_df, invariance_results