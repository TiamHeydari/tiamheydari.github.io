from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm



# ============================================================
# Basic utilities
# ============================================================

def get_device() -> torch.device:
    """
    Choose GPU if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def freeze_model(model: nn.Module) -> nn.Module:
    """
    Freeze all model parameters.

    Useful after training the encoder, before probing:

        z = model.encode(x)
    """
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model


def unfreeze_model(model: nn.Module) -> nn.Module:
    """
    Unfreeze all model parameters.
    """
    model.train()

    for param in model.parameters():
        param.requires_grad = True

    return model


# ============================================================
# Checkpoint utilities
# ============================================================

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    train_loss: Optional[float] = None,
    val_loss: Optional[float] = None,
    history: Optional[Dict[str, list]] = None,
    config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a training checkpoint.

    This saves:
        - model weights
        - optimizer state, if provided
        - epoch
        - train loss
        - validation loss
        - history
        - config
        - extra metadata

    Args:
        path: checkpoint file path
        model: PyTorch model
        optimizer: optional optimizer
        epoch: current epoch
        train_loss: training loss at save time
        val_loss: validation loss at save time
        history: training history dictionary
        config: model/training config dictionary
        extra: any extra information
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "history": history,
        "config": config,
        "extra": extra,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)
    print(f"Saved checkpoint to: {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    eval_mode: bool = True,
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], Dict[str, Any]]:
    """
    Load a training checkpoint.

    Args:
        path: checkpoint file path
        model: initialized model with the same architecture
        optimizer: optional optimizer to restore
        device: cuda/cpu
        eval_mode: if True, set model.eval()

    Returns:
        model, optimizer, checkpoint
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if eval_mode:
        model.eval()

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loaded checkpoint from: {path}")

    if checkpoint.get("epoch") is not None:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")

    if checkpoint.get("val_loss") is not None:
        print(f"Checkpoint val loss: {checkpoint['val_loss']:.5f}")

    return model, optimizer, checkpoint


def checkpoint_exists(path: str) -> bool:
    """
    Check whether a checkpoint exists.
    """
    return Path(path).exists()



# ============================================================
# Generic training history plotting
# ============================================================

import matplotlib.pyplot as plt


def plot_training_history(
    history,
    title: str = "Training history",
    metrics=None,
):
    """
    Generic training-history plotter.

    Works for:
        Autoencoder history:
            train_loss, val_loss

        β-VAE history:
            train_loss, train_recon_loss, train_kl_loss,
            val_loss, val_recon_loss, val_kl_loss

    It plots train_* and the corresponding val_* curve if available.
    """

    if history is None:
        raise ValueError(
            "history is None. This may happen if you loaded an old checkpoint "
            "that did not store history. Retrain with force_train=True."
        )

    if not isinstance(history, dict):
        raise TypeError("history should be a dictionary.")

    # If metrics not provided, automatically find train_* keys
    if metrics is None:
        train_keys = [
            k for k in history.keys()
            if k.startswith("train_") and len(history[k]) > 0
        ]

        # Put total loss first if available
        train_keys = sorted(train_keys)
        if "train_loss" in train_keys:
            train_keys.remove("train_loss")
            train_keys = ["train_loss"] + train_keys

    else:
        train_keys = [f"train_{m}" for m in metrics]

    for train_key in train_keys:
        if train_key not in history:
            continue

        train_values = history[train_key]

        if len(train_values) == 0:
            continue

        val_key = train_key.replace("train_", "val_")

        epochs = range(1, len(train_values) + 1)

        metric_name = train_key.replace("train_", "")

        plt.figure(figsize=(5, 3))

        plt.plot(
            epochs,
            train_values,
            marker="o",
            label="train",
        )

        if val_key in history and len(history[val_key]) > 0:
            val_epochs = range(1, len(history[val_key]) + 1)

            plt.plot(
                val_epochs,
                history[val_key],
                marker="o",
                label="val",
            )

        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(f"{title}: {metric_name}")
        plt.legend()
        plt.tight_layout()
        plt.show()



# ============================================================# Autoencoder loss# ============================================================
# ============================================================# Autoencoder loss# ============================================================
# ============================================================# Autoencoder loss# ============================================================

def autoencoder_loss(
    x_logits: torch.Tensor,
    x: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Reconstruction loss for binary dSprites images.

    The decoder returns logits, so we use BCEWithLogitsLoss.

    Args:
        x_logits: reconstructed image logits, shape [B, 1, 64, 64]
        x: target image, shape [B, 1, 64, 64]
        reduction: "mean" or "sum"

    Returns:
        reconstruction loss
    """
    return nn.functional.binary_cross_entropy_with_logits(
        x_logits,
        x,
        reduction=reduction,
    )


# ============================================================# One training epoch# ============================================================

def train_autoencoder_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: Optional[torch.device] = None,
) -> float:
    """
    Train autoencoder for one epoch.

    Assumes each batch is a dictionary with:
        batch["x"] -> image tensor [B, 1, 64, 64]

    Model forward:
        x_logits, z = model(x)

    Returns:
        mean training loss
    """
    if device is None:
        device = get_device()

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        x = batch["x"].to(device)

        x_logits, z = model(x)

        loss = autoencoder_loss(x_logits, x, reduction="mean")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    mean_loss = total_loss / max(total_samples, 1)
    return mean_loss


# ============================================================
# Validation epoch
# ============================================================

@torch.no_grad()
def evaluate_autoencoder(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> float:
    """
    Evaluate autoencoder reconstruction loss.

    Returns:
        mean validation loss
    """
    if device is None:
        device = get_device()

    model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        x = batch["x"].to(device)

        x_logits, z = model(x)

        loss = autoencoder_loss(x_logits, x, reduction="mean")

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    mean_loss = total_loss / max(total_samples, 1)
    return mean_loss


# ============================================================
# Full autoencoder training loop with checkpoint saving
# ============================================================

def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = "./checkpoints/best_autoencoder.pt",
    last_checkpoint_path: Optional[str] = "./checkpoints/last_autoencoder.pt",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, list]:
    """
    Full training loop for an Autoencoder.

    Saves:
        - best checkpoint based on validation loss if val_loader is provided
        - best checkpoint based on train loss if val_loader is not provided
        - optional last checkpoint at every epoch

    Args:
        model: Autoencoder with forward(x) -> x_logits, z
        train_loader: training DataLoader
        val_loader: optional validation DataLoader
        num_epochs: number of epochs
        lr: learning rate
        weight_decay: Adam weight decay
        device: cuda/cpu
        checkpoint_path: path for best checkpoint
        last_checkpoint_path: path for last checkpoint
        config: optional config dictionary

    Returns:
        history dictionary
    """
    if device is None:
        device = get_device()

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_score = float("inf")
    best_epoch = None

    for epoch in range(1, num_epochs + 1):
        train_loss = train_autoencoder_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_loss = evaluate_autoencoder(
                model=model,
                loader=val_loader,
                device=device,
            )
            history["val_loss"].append(val_loss)

            selection_score = val_loss

            print(
                f"Epoch {epoch:03d}/{num_epochs} | "
                f"train loss: {train_loss:.5f} | "
                f"val loss: {val_loss:.5f}"
            )

        else:
            val_loss = None
            selection_score = train_loss

            print(
                f"Epoch {epoch:03d}/{num_epochs} | "
                f"train loss: {train_loss:.5f}"
            )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------
        if selection_score < best_score:
            best_score = selection_score
            best_epoch = epoch

            if checkpoint_path is not None:
                save_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    history=history,
                    config=config,
                    extra={
                        "best_epoch": best_epoch,
                        "best_score": best_score,
                        "selection_metric": "val_loss" if val_loader is not None else "train_loss",
                    },
                )

        # ----------------------------------------------------
        # Save last checkpoint
        # ----------------------------------------------------
        if last_checkpoint_path is not None:
            save_checkpoint(
                path=last_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                history=history,
                config=config,
                extra={
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "selection_metric": "val_loss" if val_loader is not None else "train_loss",
                },
            )

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best score: {best_score:.5f}")

    if checkpoint_path is not None:
        print(f"Best checkpoint path: {checkpoint_path}")

    return history


# ============================================================
# Train or load helper
# ============================================================

def train_or_load_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    checkpoint_path: str = "./checkpoints/best_autoencoder.pt",
    force_train: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[nn.Module, Optional[Dict[str, list]], Optional[Dict[str, Any]]]:
    """
    If checkpoint exists, load it.
    Otherwise train and save it.

    Args:
        model: initialized Autoencoder
        train_loader: training DataLoader
        val_loader: validation DataLoader
        num_epochs: number of epochs
        lr: learning rate
        weight_decay: Adam weight decay
        device: cuda/cpu
        checkpoint_path: best checkpoint path
        force_train: if True, ignore checkpoint and train again
        config: optional config dictionary

    Returns:
        model, history, checkpoint

        If loaded:
            history comes from checkpoint if available.
        If trained:
            checkpoint is None.
    """
    if device is None:
        device = get_device()

    if checkpoint_exists(checkpoint_path) and not force_train:
        model, _, checkpoint = load_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=None,
            device=device,
            eval_mode=True,
        )

        history = checkpoint.get("history", None)
        return model, history, checkpoint

    history = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        checkpoint_path=checkpoint_path,
        last_checkpoint_path=None,
        config=config,
    )

    model, _, checkpoint = load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=None,
        device=device,
        eval_mode=True,
    )

    return model, history, checkpoint


# ============================================================
# Extract representations z = model.encode(x)
# ============================================================

@torch.no_grad()
def extract_representations(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Extract latent representations from a frozen/trained model.

    Assumes each batch contains:
        batch["x"]          -> [B, 1, 64, 64]
        batch["y"]          -> [B, 6]
        batch["y_values"]   -> [B, 6]

    Returns:
        {
            "Z": latent representations [N, D_z],
            "Y": factor classes [N, 6],
            "Y_values": factor values [N, 6],
        }
    """
    if device is None:
        device = get_device()

    model.to(device)
    model.eval()

    all_z = []
    all_y = []
    all_y_values = []

    for batch in tqdm(loader, desc="Extract z"):
        x = batch["x"].to(device)

        z = model.encode(x)

        all_z.append(z.detach().cpu())
        all_y.append(batch["y"].detach().cpu())
        all_y_values.append(batch["y_values"].detach().cpu())

    Z = torch.cat(all_z, dim=0)
    Y = torch.cat(all_y, dim=0)
    Y_values = torch.cat(all_y_values, dim=0)

    return {
        "Z": Z,
        "Y": Y,
        "Y_values": Y_values,
    }


# ============================================================
# Reconstruction helper
# ============================================================

@torch.no_grad()
def reconstruct_batch(
    model: nn.Module,
    x: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Reconstruct a batch of images.

    Args:
        model: autoencoder
        x: input image tensor [B, 1, 64, 64]

    Returns:
        x_recon: reconstructed image probabilities in [0, 1]
    """
    if device is None:
        device = get_device()

    model.to(device)
    model.eval()

    x = x.to(device)

    x_logits, z = model(x)

    x_recon = torch.sigmoid(x_logits)

    return x_recon.detach().cpu()





# ============================================================# Beta-VAE training# ============================================================
# ============================================================# Beta-VAE training# ============================================================
# ============================================================# Beta-VAE training# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def beta_vae_loss(
    x_logits: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 4.0,
) -> Dict[str, torch.Tensor]:
    """
    β-VAE loss.

    Reconstruction:
        BCEWithLogits(x_logits, x)

    KL:
        KL(q(z|x) || p(z))

    Total:
        L = reconstruction + beta * KL
    """

    recon_loss = F.binary_cross_entropy_with_logits(
        x_logits,
        x,
        reduction="none",
    )

    # Sum over pixels, then average over batch
    recon_loss = recon_loss.view(x.shape[0], -1).sum(dim=1).mean()

    # KL per sample, then average over batch
    kl_loss = -0.5 * torch.sum(
        1.0 + logvar - mu.pow(2) - logvar.exp(),
        dim=1,
    ).mean()

    total_loss = recon_loss + beta * kl_loss

    return {
        "loss": total_loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
    }


def train_beta_vae_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    beta: float = 4.0,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Train β-VAE for one epoch.
    """
    if device is None:
        device = get_device()

    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc="Train β-VAE", leave=False):
        x = batch["x"].to(device)

        x_logits, mu, logvar, z = model(x)

        losses = beta_vae_loss(
            x_logits=x_logits,
            x=x,
            mu=mu,
            logvar=logvar,
            beta=beta,
        )

        loss = losses["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = x.shape[0]

        total_loss += losses["loss"].item() * batch_size
        total_recon += losses["recon_loss"].item() * batch_size
        total_kl += losses["kl_loss"].item() * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "recon_loss": total_recon / max(total_samples, 1),
        "kl_loss": total_kl / max(total_samples, 1),
    }


@torch.no_grad()
def evaluate_beta_vae(
    model: nn.Module,
    loader: DataLoader,
    beta: float = 4.0,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate β-VAE.
    """
    if device is None:
        device = get_device()

    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc="Val β-VAE", leave=False):
        x = batch["x"].to(device)

        x_logits, mu, logvar, z = model(x)

        losses = beta_vae_loss(
            x_logits=x_logits,
            x=x,
            mu=mu,
            logvar=logvar,
            beta=beta,
        )

        batch_size = x.shape[0]

        total_loss += losses["loss"].item() * batch_size
        total_recon += losses["recon_loss"].item() * batch_size
        total_kl += losses["kl_loss"].item() * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "recon_loss": total_recon / max(total_samples, 1),
        "kl_loss": total_kl / max(total_samples, 1),
    }


def train_beta_vae(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 20,
    lr: float = 1e-3,
    beta: float = 4.0,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = "./checkpoints/best_beta_vae.pt",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, list]:
    """
    Full β-VAE training loop.

    Saves the best model based on validation total loss.
    """
    if device is None:
        device = get_device()

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    history = {
        "train_loss": [],
        "train_recon_loss": [],
        "train_kl_loss": [],
        "val_loss": [],
        "val_recon_loss": [],
        "val_kl_loss": [],
    }

    best_score = float("inf")
    best_epoch = None

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_beta_vae_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            beta=beta,
            device=device,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_recon_loss"].append(train_metrics["recon_loss"])
        history["train_kl_loss"].append(train_metrics["kl_loss"])

        if val_loader is not None:
            val_metrics = evaluate_beta_vae(
                model=model,
                loader=val_loader,
                beta=beta,
                device=device,
            )

            history["val_loss"].append(val_metrics["loss"])
            history["val_recon_loss"].append(val_metrics["recon_loss"])
            history["val_kl_loss"].append(val_metrics["kl_loss"])

            selection_score = val_metrics["loss"]

            print(
                f"Epoch {epoch:03d}/{num_epochs} | "
                f"train loss: {train_metrics['loss']:.3f} | "
                f"train recon: {train_metrics['recon_loss']:.3f} | "
                f"train KL: {train_metrics['kl_loss']:.3f} | "
                f"val loss: {val_metrics['loss']:.3f} | "
                f"val recon: {val_metrics['recon_loss']:.3f} | "
                f"val KL: {val_metrics['kl_loss']:.3f}"
            )

        else:
            val_metrics = None
            selection_score = train_metrics["loss"]

            print(
                f"Epoch {epoch:03d}/{num_epochs} | "
                f"train loss: {train_metrics['loss']:.3f} | "
                f"train recon: {train_metrics['recon_loss']:.3f} | "
                f"train KL: {train_metrics['kl_loss']:.3f}"
            )

        if selection_score < best_score:
            best_score = selection_score
            best_epoch = epoch

            if checkpoint_path is not None:
                save_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_loss=train_metrics["loss"],
                    val_loss=val_metrics["loss"] if val_metrics is not None else None,
                    history=history,
                    config=config,
                    extra={
                        "model_type": "BetaVAE",
                        "beta": beta,
                        "best_epoch": best_epoch,
                        "best_score": best_score,
                        "selection_metric": "val_loss" if val_loader is not None else "train_loss",
                    },
                )

    print("\nβ-VAE training complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best score: {best_score:.3f}")

    return history


def train_or_load_beta_vae(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 20,
    lr: float = 1e-3,
    beta: float = 4.0,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    checkpoint_path: str = "./checkpoints/best_beta_vae.pt",
    force_train: bool = False,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Load β-VAE checkpoint if it exists.
    Otherwise train β-VAE and save the best checkpoint.
    """
    if device is None:
        device = get_device()

    if checkpoint_exists(checkpoint_path) and not force_train:
        model, _, checkpoint = load_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=None,
            device=device,
            eval_mode=True,
        )

        history = checkpoint.get("history", None)
        return model, history, checkpoint

    history = train_beta_vae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=lr,
        beta=beta,
        weight_decay=weight_decay,
        device=device,
        checkpoint_path=checkpoint_path,
        config=config,
    )

    model, _, checkpoint = load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=None,
        device=device,
        eval_mode=True,
    )

    return model, history, checkpoint


@torch.no_grad()
def reconstruct_batch_beta_vae(
    model: nn.Module,
    x: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Reconstruct images from β-VAE using deterministic z = mu.

    Returns:
        reconstructed images in [0, 1]
    """
    if device is None:
        device = get_device()

    model.to(device)
    model.eval()

    x = x.to(device)

    z = model.encode(x)
    x_logits = model.decode(z)
    x_recon = torch.sigmoid(x_logits)

    return x_recon.detach().cpu()








# ============================================================# β-TCVAE training# ============================================================
# ============================================================# β-TCVAE training# ============================================================
# ============================================================# β-TCVAE training# ============================================================

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def log_density_gaussian(
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """
    Compute element-wise log density of a Gaussian.

    log N(x | mu, exp(logvar))

    Supports broadcasting.

    Returns:
        log density with same broadcasted shape as x, mu, logvar.
    """
    log_2pi = math.log(2.0 * math.pi)

    return -0.5 * (
        log_2pi
        + logvar
        + ((x - mu) ** 2) * torch.exp(-logvar)
    )


def estimate_tcvae_decomposition(
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Estimate the β-TCVAE KL decomposition using minibatch estimates.

    We decompose:

        KL(q(z|x) || p(z))
        =
        I_q(x;z)
        + TC(z)
        + sum_j KL(q(z_j) || p(z_j))

    where:

        MI:
            I_q(x;z) = E[log q(z|x) - log q(z)]

        TC:
            TC(z) = E[log q(z) - sum_j log q(z_j)]

        Dimension-wise KL:
            DW-KL = E[sum_j log q(z_j) - log p(z)]

    Args:
        z:
            sampled latent, [B, D_z]

        mu:
            encoder mean, [B, D_z]

        logvar:
            encoder log variance, [B, D_z]

    Returns:
        mi_loss, tc_loss, dw_kl_loss
    """
    B, D_z = z.shape

    # --------------------------------------------------------
    # log q(z_i | x_i)
    # shape: [B]
    # --------------------------------------------------------
    log_q_z_given_x = log_density_gaussian(
        z,
        mu,
        logvar,
    ).sum(dim=1)

    # --------------------------------------------------------
    # Matrix of log q(z_i | x_j)
    #
    # z_i:  [B, 1, D_z]
    # mu_j: [1, B, D_z]
    #
    # result: [B, B, D_z]
    # --------------------------------------------------------
    z_i = z.unsqueeze(1)
    mu_j = mu.unsqueeze(0)
    logvar_j = logvar.unsqueeze(0)

    log_q_zCx_matrix = log_density_gaussian(
        z_i,
        mu_j,
        logvar_j,
    )  # [B, B, D_z]

    # --------------------------------------------------------
    # log q(z)
    #
    # q(z_i) ≈ 1/B sum_j q(z_i | x_j)
    # --------------------------------------------------------
    log_q_z = torch.logsumexp(
        log_q_zCx_matrix.sum(dim=2),
        dim=1,
    ) - math.log(B)

    # --------------------------------------------------------
    # sum_j log q(z_j)
    #
    # For each latent dimension d:
    #     q(z_i,d) ≈ 1/B sum_j q(z_i,d | x_j)
    # --------------------------------------------------------
    log_prod_q_z = torch.logsumexp(
        log_q_zCx_matrix,
        dim=1,
    ).sum(dim=1) - math.log(B) * D_z

    # --------------------------------------------------------
    # log p(z), where p(z) = N(0, I)
    # --------------------------------------------------------
    zeros = torch.zeros_like(z)

    log_p_z = log_density_gaussian(
        z,
        zeros,
        zeros,
    ).sum(dim=1)

    # --------------------------------------------------------
    # Decomposition
    # --------------------------------------------------------
    mi_loss = (log_q_z_given_x - log_q_z).mean()

    tc_loss = (log_q_z - log_prod_q_z).mean()

    dw_kl_loss = (log_prod_q_z - log_p_z).mean()

    return {
        "mi_loss": mi_loss,
        "tc_loss": tc_loss,
        "dw_kl_loss": dw_kl_loss,
    }


def beta_tcvae_loss(
    x_logits: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    alpha_mi: float = 1.0,
    beta_tc: float = 4.0,
    gamma_dwkl: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    β-TCVAE loss.

    Total loss:

        L = recon
            + alpha * MI
            + beta  * TC
            + gamma * DW-KL

    The key disentanglement pressure is beta_tc * TC.
    """

    # --------------------------------------------------------
    # Reconstruction loss
    # --------------------------------------------------------
    recon_loss = F.binary_cross_entropy_with_logits(
        x_logits,
        x,
        reduction="none",
    )

    recon_loss = recon_loss.view(x.shape[0], -1).sum(dim=1).mean()

    # --------------------------------------------------------
    # KL decomposition
    # --------------------------------------------------------
    terms = estimate_tcvae_decomposition(
        z=z,
        mu=mu,
        logvar=logvar,
    )

    mi_loss = terms["mi_loss"]
    tc_loss = terms["tc_loss"]
    dw_kl_loss = terms["dw_kl_loss"]

    total_loss = (
        recon_loss
        + alpha_mi * mi_loss
        + beta_tc * tc_loss
        + gamma_dwkl * dw_kl_loss
    )

    return {
        "loss": total_loss,
        "recon_loss": recon_loss,
        "mi_loss": mi_loss,
        "tc_loss": tc_loss,
        "dw_kl_loss": dw_kl_loss,
    }


def train_beta_tcvae_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    alpha_mi: float = 1.0,
    beta_tc: float = 4.0,
    gamma_dwkl: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Train β-TCVAE for one epoch.
    """
    if device is None:
        device = get_device()

    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_mi = 0.0
    total_tc = 0.0
    total_dwkl = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc="Train β-TCVAE", leave=False):
        x = batch["x"].to(device)

        x_logits, mu, logvar, z = model(x)

        losses = beta_tcvae_loss(
            x_logits=x_logits,
            x=x,
            z=z,
            mu=mu,
            logvar=logvar,
            alpha_mi=alpha_mi,
            beta_tc=beta_tc,
            gamma_dwkl=gamma_dwkl,
        )

        loss = losses["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = x.shape[0]

        total_loss += losses["loss"].item() * batch_size
        total_recon += losses["recon_loss"].item() * batch_size
        total_mi += losses["mi_loss"].item() * batch_size
        total_tc += losses["tc_loss"].item() * batch_size
        total_dwkl += losses["dw_kl_loss"].item() * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "recon_loss": total_recon / max(total_samples, 1),
        "mi_loss": total_mi / max(total_samples, 1),
        "tc_loss": total_tc / max(total_samples, 1),
        "dw_kl_loss": total_dwkl / max(total_samples, 1),
    }


@torch.no_grad()
def evaluate_beta_tcvae(
    model: nn.Module,
    loader: DataLoader,
    alpha_mi: float = 1.0,
    beta_tc: float = 4.0,
    gamma_dwkl: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate β-TCVAE.
    """
    if device is None:
        device = get_device()

    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_mi = 0.0
    total_tc = 0.0
    total_dwkl = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc="Val β-TCVAE", leave=False):
        x = batch["x"].to(device)

        x_logits, mu, logvar, z = model(x)

        losses = beta_tcvae_loss(
            x_logits=x_logits,
            x=x,
            z=z,
            mu=mu,
            logvar=logvar,
            alpha_mi=alpha_mi,
            beta_tc=beta_tc,
            gamma_dwkl=gamma_dwkl,
        )

        batch_size = x.shape[0]

        total_loss += losses["loss"].item() * batch_size
        total_recon += losses["recon_loss"].item() * batch_size
        total_mi += losses["mi_loss"].item() * batch_size
        total_tc += losses["tc_loss"].item() * batch_size
        total_dwkl += losses["dw_kl_loss"].item() * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "recon_loss": total_recon / max(total_samples, 1),
        "mi_loss": total_mi / max(total_samples, 1),
        "tc_loss": total_tc / max(total_samples, 1),
        "dw_kl_loss": total_dwkl / max(total_samples, 1),
    }


def train_beta_tcvae(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 20,
    lr: float = 1e-3,
    alpha_mi: float = 1.0,
    beta_tc: float = 4.0,
    gamma_dwkl: float = 1.0,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = "./checkpoints/best_beta_tcvae.pt",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, list]:
    """
    Full β-TCVAE training loop.

    Saves best model based on validation total loss.
    """
    if device is None:
        device = get_device()

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    history = {
        "train_loss": [],
        "train_recon_loss": [],
        "train_mi_loss": [],
        "train_tc_loss": [],
        "train_dw_kl_loss": [],
        "val_loss": [],
        "val_recon_loss": [],
        "val_mi_loss": [],
        "val_tc_loss": [],
        "val_dw_kl_loss": [],
    }

    best_score = float("inf")
    best_epoch = None

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_beta_tcvae_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            alpha_mi=alpha_mi,
            beta_tc=beta_tc,
            gamma_dwkl=gamma_dwkl,
            device=device,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_recon_loss"].append(train_metrics["recon_loss"])
        history["train_mi_loss"].append(train_metrics["mi_loss"])
        history["train_tc_loss"].append(train_metrics["tc_loss"])
        history["train_dw_kl_loss"].append(train_metrics["dw_kl_loss"])

        if val_loader is not None:
            val_metrics = evaluate_beta_tcvae(
                model=model,
                loader=val_loader,
                alpha_mi=alpha_mi,
                beta_tc=beta_tc,
                gamma_dwkl=gamma_dwkl,
                device=device,
            )

            history["val_loss"].append(val_metrics["loss"])
            history["val_recon_loss"].append(val_metrics["recon_loss"])
            history["val_mi_loss"].append(val_metrics["mi_loss"])
            history["val_tc_loss"].append(val_metrics["tc_loss"])
            history["val_dw_kl_loss"].append(val_metrics["dw_kl_loss"])

            selection_score = val_metrics["loss"]

            print(
                f"Epoch {epoch:03d}/{num_epochs} | "
                f"train loss: {train_metrics['loss']:.3f} | "
                f"recon: {train_metrics['recon_loss']:.3f} | "
                f"MI: {train_metrics['mi_loss']:.3f} | "
                f"TC: {train_metrics['tc_loss']:.3f} | "
                f"DW-KL: {train_metrics['dw_kl_loss']:.3f} | "
                f"val loss: {val_metrics['loss']:.3f} | "
                f"val recon: {val_metrics['recon_loss']:.3f} | "
                f"val TC: {val_metrics['tc_loss']:.3f}"
            )

        else:
            val_metrics = None
            selection_score = train_metrics["loss"]

            print(
                f"Epoch {epoch:03d}/{num_epochs} | "
                f"train loss: {train_metrics['loss']:.3f} | "
                f"recon: {train_metrics['recon_loss']:.3f} | "
                f"MI: {train_metrics['mi_loss']:.3f} | "
                f"TC: {train_metrics['tc_loss']:.3f} | "
                f"DW-KL: {train_metrics['dw_kl_loss']:.3f}"
            )

        if selection_score < best_score:
            best_score = selection_score
            best_epoch = epoch

            if checkpoint_path is not None:
                save_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_loss=train_metrics["loss"],
                    val_loss=val_metrics["loss"] if val_metrics is not None else None,
                    history=history,
                    config=config,
                    extra={
                        "model_type": "BetaTCVAE",
                        "alpha_mi": alpha_mi,
                        "beta_tc": beta_tc,
                        "gamma_dwkl": gamma_dwkl,
                        "best_epoch": best_epoch,
                        "best_score": best_score,
                        "selection_metric": "val_loss" if val_loader is not None else "train_loss",
                    },
                )

    print("\nβ-TCVAE training complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best score: {best_score:.3f}")

    return history


def train_or_load_beta_tcvae(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 20,
    lr: float = 1e-3,
    alpha_mi: float = 1.0,
    beta_tc: float = 4.0,
    gamma_dwkl: float = 1.0,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    checkpoint_path: str = "./checkpoints/best_beta_tcvae.pt",
    force_train: bool = False,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Load β-TCVAE checkpoint if it exists.
    Otherwise train β-TCVAE and save the best checkpoint.
    """
    if device is None:
        device = get_device()

    if checkpoint_exists(checkpoint_path) and not force_train:
        model, _, checkpoint = load_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=None,
            device=device,
            eval_mode=True,
        )

        history = checkpoint.get("history", None)

        return model, history, checkpoint

    history = train_beta_tcvae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=lr,
        alpha_mi=alpha_mi,
        beta_tc=beta_tc,
        gamma_dwkl=gamma_dwkl,
        weight_decay=weight_decay,
        device=device,
        checkpoint_path=checkpoint_path,
        config=config,
    )

    model, _, checkpoint = load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=None,
        device=device,
        eval_mode=True,
    )

    return model, history, checkpoint


@torch.no_grad()
def reconstruct_batch_beta_tcvae(
    model: nn.Module,
    x: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Reconstruct images from β-TCVAE using deterministic z = mu.

    Returns:
        reconstructed images in [0, 1]
    """
    if device is None:
        device = get_device()

    model.to(device)
    model.eval()

    x = x.to(device)

    z = model.encode(x)      # z = mu
    x_logits = model.decode(z)
    x_recon = torch.sigmoid(x_logits)

    return x_recon.detach().cpu()







