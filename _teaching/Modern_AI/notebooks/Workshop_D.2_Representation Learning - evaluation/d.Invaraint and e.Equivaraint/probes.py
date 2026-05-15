# ============================================================
# probes.py
# Probe utilities for representation evaluation
# ============================================================

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

try:
    import pandas as pd
except ImportError:
    pd = None


# ------------------------------------------------------------
# Device helper
# ------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# Linear / small MLP probe
# ------------------------------------------------------------

class FactorProbe(nn.Module):
    """
    Small probe g_phi(z) -> factor label.

    By default this is a linear probe.
    If hidden_dim is not None, it becomes a small MLP.
    """

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()

        if hidden_dim is None:
            self.net = nn.Linear(latent_dim, num_classes)
        else:
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ------------------------------------------------------------
# DataLoader from Z and labels
# ------------------------------------------------------------

def make_probe_loader(
    Z: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 256,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create DataLoader for probe training.

    Z: [N, D_z]
    y: [N]
    """
    dataset = TensorDataset(Z.float(), y.long())

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )

    return loader


# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------

@torch.no_grad()
def compute_accuracy(
    logits: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """
    Classification accuracy.
    """
    pred = logits.argmax(dim=1)
    acc = (pred == y).float().mean().item()
    return acc


# ------------------------------------------------------------
# Train one epoch
# ------------------------------------------------------------

def train_probe_epoch(
    probe: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """
    Train probe for one epoch.

    Returns:
        mean_loss, mean_accuracy
    """
    if device is None:
        device = get_device()

    probe.train()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for z, y in tqdm(loader, desc="Train probe", leave=False):
        z = z.to(device)
        y = y.to(device)

        logits = probe(z)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = z.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += batch_size

    mean_loss = total_loss / max(total_samples, 1)
    mean_acc = total_correct / max(total_samples, 1)

    return mean_loss, mean_acc


# ------------------------------------------------------------
# Evaluate probe
# ------------------------------------------------------------

@torch.no_grad()
def evaluate_probe(
    probe: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """
    Evaluate probe.

    Returns:
        mean_loss, mean_accuracy
    """
    if device is None:
        device = get_device()

    probe.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for z, y in tqdm(loader, desc="Eval probe", leave=False):
        z = z.to(device)
        y = y.to(device)

        logits = probe(z)
        loss = criterion(logits, y)

        batch_size = z.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += batch_size

    mean_loss = total_loss / max(total_samples, 1)
    mean_acc = total_correct / max(total_samples, 1)

    return mean_loss, mean_acc


# ------------------------------------------------------------
# Train one factor probe
# ------------------------------------------------------------

def train_factor_probe(
    Z_train: torch.Tensor,
    y_train: torch.Tensor,
    Z_val: torch.Tensor,
    y_val: torch.Tensor,
    num_classes: int,
    hidden_dim: Optional[int] = None,
    num_epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Train one probe for one factor.

    Example:
        g_shape(z) -> shape label
    """
    if device is None:
        device = get_device()

    latent_dim = Z_train.shape[1]

    probe = FactorProbe(
        latent_dim=latent_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    train_loader = make_probe_loader(
        Z=Z_train,
        y=y_train,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = make_probe_loader(
        Z=Z_val,
        y=y_val,
        batch_size=batch_size,
        shuffle=False,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_state_dict = None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_probe_epoch(
            probe=probe,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = evaluate_probe(
            probe=probe,
            loader=val_loader,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"train acc: {train_acc:.4f} | "
            f"val acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in probe.state_dict().items()
            }

    if best_state_dict is not None:
        probe.load_state_dict(best_state_dict)

    return probe, history


# ------------------------------------------------------------
# Run probes for all factors
# ------------------------------------------------------------

def run_all_factor_probes(
    Z_train: torch.Tensor,
    Y_train: torch.Tensor,
    Z_val: torch.Tensor,
    Y_val: torch.Tensor,
    Z_test: torch.Tensor,
    Y_test: torch.Tensor,
    factor_names: List[str],
    skip_factors: Tuple[str, ...] = ("color",),
    hidden_dim: Optional[int] = None,
    num_epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
):
    """
    Train one probe per factor.

    Uses:
        Z_train -> train probe
        Z_val   -> select best probe
        Z_test  -> final reported score

    Returns:
        results_df, probe_models, histories
    """
    if device is None:
        device = get_device()

    results = []
    probe_models = {}
    histories = {}

    for factor_idx, factor_name in enumerate(factor_names):
        if factor_name in skip_factors:
            print(f"\nSkipping factor: {factor_name}")
            continue

        print("\n" + "=" * 70)
        print(f"Training probe for factor: {factor_name}")
        print("=" * 70)

        y_train = Y_train[:, factor_idx].long()
        y_val = Y_val[:, factor_idx].long()
        y_test = Y_test[:, factor_idx].long()

        all_y = torch.cat([y_train, y_val, y_test], dim=0)
        num_classes = int(all_y.max().item()) + 1

        probe, history = train_factor_probe(
            Z_train=Z_train,
            y_train=y_train,
            Z_val=Z_val,
            y_val=y_val,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_epochs=num_epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
        )

        test_loader = make_probe_loader(
            Z=Z_test,
            y=y_test,
            batch_size=batch_size,
            shuffle=False,
        )

        test_loss, test_acc = evaluate_probe(
            probe=probe,
            loader=test_loader,
            device=device,
        )

        print(f"Final test accuracy for {factor_name}: {test_acc:.4f}")

        probe_models[factor_name] = probe
        histories[factor_name] = history

        results.append(
            {
                "factor": factor_name,
                "num_classes": num_classes,
                "best_val_acc": max(history["val_acc"]),
                "test_acc": test_acc,
                "test_loss": test_loss,
            }
        )

    if pd is not None:
        results_df = pd.DataFrame(results)
    else:
        results_df = results

    return results_df, probe_models, histories













# ============================================================
# Single-latent-dimension probes
# For heatmap: latent dimension z_m vs. factor FV_k
# ============================================================

from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


class OneDimFactorProbe(nn.Module):
    """
    Probe from one latent dimension z_m to one factor label.

    g_{phi_{k,m}}: R^1 -> R^{C_k}
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(1, num_classes)

    def forward(self, z_m: torch.Tensor) -> torch.Tensor:
        return self.linear(z_m)


def make_1d_probe_loader(
    z_m: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 256,
    shuffle: bool = True,
) -> DataLoader:
    """
    DataLoader for one-dimensional probe training.

    z_m: [N, 1]
    y:   [N]
    """
    dataset = TensorDataset(z_m.float(), y.long())

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )

    return loader


def standardize_1d_using_train(
    z_train_m: torch.Tensor,
    z_val_m: torch.Tensor,
    z_test_m: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standardize one latent dimension using train statistics only.
    """
    mean = z_train_m.mean()
    std = z_train_m.std() + 1e-8

    z_train_m = (z_train_m - mean) / std
    z_val_m = (z_val_m - mean) / std
    z_test_m = (z_test_m - mean) / std

    return z_train_m, z_val_m, z_test_m


@torch.no_grad()
def evaluate_1d_factor_probe(
    probe: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> float:
    """
    Return classification accuracy for one-dimensional probe.
    """
    if device is None:
        device = get_device()

    probe.eval()

    total_correct = 0
    total_samples = 0

    for z_m, y in loader:
        z_m = z_m.to(device)
        y = y.to(device)

        logits = probe(z_m)
        pred = logits.argmax(dim=1)

        total_correct += (pred == y).sum().item()
        total_samples += y.shape[0]

    acc = total_correct / max(total_samples, 1)
    return acc


def train_1d_factor_probe(
    z_train_m: torch.Tensor,
    y_train: torch.Tensor,
    z_val_m: torch.Tensor,
    y_val: torch.Tensor,
    z_test_m: torch.Tensor,
    y_test: torch.Tensor,
    num_classes: int,
    num_epochs: int = 10,
    lr: float = 1e-2,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Train one probe:

        g_{phi_{k,m}}(z_m) ≈ FV_k(x)

    This is for one latent dimension m and one factor k.

    Returns:
        best_val_acc, test_acc, chance_acc, normalized_test_acc
    """
    if device is None:
        device = get_device()

    z_train_m, z_val_m, z_test_m = standardize_1d_using_train(
        z_train_m=z_train_m,
        z_val_m=z_val_m,
        z_test_m=z_test_m,
    )

    train_loader = make_1d_probe_loader(
        z_m=z_train_m,
        y=y_train,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = make_1d_probe_loader(
        z_m=z_val_m,
        y=y_val,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = make_1d_probe_loader(
        z_m=z_test_m,
        y=y_test,
        batch_size=batch_size,
        shuffle=False,
    )

    probe = OneDimFactorProbe(num_classes=num_classes).to(device)

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state_dict = None

    for epoch in range(num_epochs):
        probe.train()

        for z_batch, y_batch in train_loader:
            z_batch = z_batch.to(device)
            y_batch = y_batch.to(device)

            logits = probe(z_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_acc = evaluate_1d_factor_probe(
            probe=probe,
            loader=val_loader,
            device=device,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in probe.state_dict().items()
            }

    if best_state_dict is not None:
        probe.load_state_dict(best_state_dict)

    test_acc = evaluate_1d_factor_probe(
        probe=probe,
        loader=test_loader,
        device=device,
    )

    chance_acc = 1.0 / num_classes

    normalized_test_acc = (
        (test_acc - chance_acc)
        / (1.0 - chance_acc)
    )

    return {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "chance_acc": chance_acc,
        "normalized_test_acc": normalized_test_acc,
    }


def compute_latent_factor_heatmap(
    Z_train: torch.Tensor,
    Y_train: torch.Tensor,
    Z_val: torch.Tensor,
    Y_val: torch.Tensor,
    Z_test: torch.Tensor,
    Y_test: torch.Tensor,
    factor_names: List[str],
    skip_factors: Tuple[str, ...] = ("color",),
    num_epochs: int = 10,
    lr: float = 1e-2,
    batch_size: int = 256,
    score_type: str = "normalized_test_acc",
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    """
    Compute heatmap:

        H_{k,m} = score of g_{phi_{k,m}}(z_m) -> FV_k(x)

    Rows:
        factors FV_k

    Columns:
        latent dimensions z_m

    score_type options:
        "test_acc"
        "normalized_test_acc"

    Returns:
        H: tensor [num_factors, latent_dim]
        kept_factor_names: list of factor names
        records: list of dictionaries with all scores
    """
    if device is None:
        device = get_device()

    if score_type not in ["test_acc", "normalized_test_acc"]:
        raise ValueError(
            "score_type must be either 'test_acc' or 'normalized_test_acc'"
        )

    latent_dim = Z_train.shape[1]

    kept_factor_names = []
    kept_factor_indices = []

    for factor_idx, factor_name in enumerate(factor_names):
        if factor_name not in skip_factors:
            kept_factor_names.append(factor_name)
            kept_factor_indices.append(factor_idx)

    H = torch.zeros(len(kept_factor_names), latent_dim)
    records = []

    for row_idx, factor_idx in enumerate(kept_factor_indices):
        factor_name = factor_names[factor_idx]

        if verbose:
            print("\n" + "=" * 70)
            print(f"Factor: {factor_name}")
            print("=" * 70)

        y_train = Y_train[:, factor_idx].long()
        y_val = Y_val[:, factor_idx].long()
        y_test = Y_test[:, factor_idx].long()

        all_y = torch.cat([y_train, y_val, y_test], dim=0)
        num_classes = int(all_y.max().item()) + 1

        for m in range(latent_dim):
            z_train_m = Z_train[:, m:m + 1]
            z_val_m = Z_val[:, m:m + 1]
            z_test_m = Z_test[:, m:m + 1]

            scores = train_1d_factor_probe(
                z_train_m=z_train_m,
                y_train=y_train,
                z_val_m=z_val_m,
                y_val=y_val,
                z_test_m=z_test_m,
                y_test=y_test,
                num_classes=num_classes,
                num_epochs=num_epochs,
                lr=lr,
                batch_size=batch_size,
                device=device,
            )

            H[row_idx, m] = scores[score_type]

            record = {
                "factor": factor_name,
                "factor_idx": factor_idx,
                "latent_dim": m,
                "num_classes": num_classes,
                **scores,
            }

            records.append(record)

            if verbose:
                print(
                    f"z_{m:02d} -> {factor_name:12s} | "
                    f"test_acc = {scores['test_acc']:.4f} | "
                    f"norm_acc = {scores['normalized_test_acc']:.4f}"
                )

    return H, kept_factor_names, records


def plot_latent_factor_heatmap(
    H: torch.Tensor,
    factor_names: List[str],
    title: str = "Latent dimension vs factor",
    score_label: str = "Probe score",
    figsize: Optional[Tuple[float, float]] = None,
):
    """
    Plot heatmap of latent dimension vs. factor score.

    H: [num_factors, latent_dim]
    """
    H_np = H.detach().cpu().numpy()

    num_factors, latent_dim = H_np.shape

    if figsize is None:
        figsize = (max(8, 0.7 * latent_dim), max(3, 0.7 * num_factors + 1.5))

    plt.figure(figsize=figsize)

    im = plt.imshow(H_np, aspect="auto")

    plt.colorbar(im, label=score_label)

    plt.yticks(
        ticks=range(num_factors),
        labels=factor_names,
    )

    plt.xticks(
        ticks=range(latent_dim),
        labels=[f"z{m}" for m in range(latent_dim)],
        rotation=90,
    )

    plt.xlabel("Latent dimension")
    plt.ylabel("Factor of variation")
    plt.title(title)

    plt.tight_layout()
    plt.show()


def run_latent_factor_heatmap(
    Z_train: torch.Tensor,
    Y_train: torch.Tensor,
    Z_val: torch.Tensor,
    Y_val: torch.Tensor,
    Z_test: torch.Tensor,
    Y_test: torch.Tensor,
    factor_names: List[str],
    model_name: str = "Model",
    skip_factors: Tuple[str, ...] = ("color",),
    num_epochs: int = 10,
    lr: float = 1e-2,
    batch_size: int = 256,
    score_type: str = "normalized_test_acc",
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    """
    Convenience wrapper:
        1. compute heatmap
        2. plot heatmap
        3. return H, factor names, records
    """
    H, kept_factor_names, records = compute_latent_factor_heatmap(
        Z_train=Z_train,
        Y_train=Y_train,
        Z_val=Z_val,
        Y_val=Y_val,
        Z_test=Z_test,
        Y_test=Y_test,
        factor_names=factor_names,
        skip_factors=skip_factors,
        num_epochs=num_epochs,
        lr=lr,
        batch_size=batch_size,
        score_type=score_type,
        device=device,
        verbose=verbose,
    )

    if score_type == "normalized_test_acc":
        score_label = "Normalized probe accuracy"
    else:
        score_label = "Probe test accuracy"

    plot_latent_factor_heatmap(
        H=H,
        factor_names=kept_factor_names,
        title=f"{model_name}: latent dimension vs factor",
        score_label=score_label,
    )

    return H, kept_factor_names, records





# ============================================================
# Clustered heatmap with dendrogram
# Cluster latent dimensions based on their factor profiles
# ============================================================

from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import (
    linkage,
    dendrogram,
    optimal_leaf_ordering,
)


def cluster_latent_dimensions(
    H: torch.Tensor,
    method: str = "ward",
    metric: str = "euclidean",
):
    """
    Cluster latent dimensions using their factor-score profiles.

    H: [num_factors, latent_dim]

    Each latent dimension z_m is represented by a vector:

        v_m = H[:, m]

    Returns:
        dim_order: ordered latent dimension indices
        Z_linkage: hierarchical clustering linkage matrix
    """
    H_np = H.detach().cpu().numpy()

    # Columns are latent dimensions, rows are factors.
    # For clustering dimensions, we transpose:
    # X_dim_profiles: [latent_dim, num_factors]
    X_dim_profiles = H_np.T

    distances = pdist(X_dim_profiles, metric=metric)

    Z_linkage = linkage(distances, method=method)

    # Improve visual ordering of leaves
    Z_linkage = optimal_leaf_ordering(Z_linkage, distances)

    dendro = dendrogram(Z_linkage, no_plot=True)
    dim_order = dendro["leaves"]

    return dim_order, Z_linkage


def plot_clustered_latent_factor_heatmap(
    H: torch.Tensor,
    factor_names: List[str],
    title: str = "Clustered latent dimension vs factor heatmap",
    score_label: str = "Probe score",
    method: str = "ward",
    metric: str = "euclidean",
    figsize: Optional[Tuple[float, float]] = None,
):
    """
    Plot dendrogram-ordered latent-factor heatmap.

    Rows:
        factors FV_k

    Columns:
        latent dimensions z_m, ordered by hierarchical clustering

    Clustering is applied only to latent dimensions.
    """
    H_np = H.detach().cpu().numpy()

    num_factors, latent_dim = H_np.shape

    dim_order, Z_linkage = cluster_latent_dimensions(
        H=H,
        method=method,
        metric=metric,
    )

    H_ordered = H_np[:, dim_order]

    if figsize is None:
        figsize = (
            max(9, 0.75 * latent_dim),
            max(5, 0.8 * num_factors + 3),
        )

    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        nrows=2,
        ncols=1,
        height_ratios=[1.2, 4.0],
        hspace=0.05,
    )

    # --------------------------------------------------------
    # Dendrogram
    # --------------------------------------------------------
    ax_dendro = fig.add_subplot(gs[0])

    dendrogram(
        Z_linkage,
        ax=ax_dendro,
        labels=[f"z{m}" for m in range(latent_dim)],
        leaf_rotation=90,
        leaf_font_size=9,
    )

    ax_dendro.set_ylabel("Distance")
    ax_dendro.set_xticks([])
    ax_dendro.set_title(title)

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------
    ax_heatmap = fig.add_subplot(gs[1])

    im = ax_heatmap.imshow(
        H_ordered,
        aspect="auto",
    )

    cbar = plt.colorbar(im, ax=ax_heatmap)
    cbar.set_label(score_label)

    ax_heatmap.set_yticks(range(num_factors))
    ax_heatmap.set_yticklabels(factor_names)

    ax_heatmap.set_xticks(range(latent_dim))
    ax_heatmap.set_xticklabels(
        [f"z{m}" for m in dim_order],
        rotation=90,
    )

    ax_heatmap.set_xlabel("Latent dimension, clustered")
    ax_heatmap.set_ylabel("Factor of variation")

    plt.tight_layout()
    plt.show()

    return {
        "H_ordered": H_ordered,
        "dim_order": dim_order,
        "linkage": Z_linkage,
    }