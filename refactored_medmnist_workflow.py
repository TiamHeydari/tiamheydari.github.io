# refactored_medmnist_workflow.py

from __future__ import annotations

import copy
import math
import random
import subprocess
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms


# =========================
# Optional dependency setup
# =========================
def ensure_package(package_name: str) -> None:
    try:
        __import__(package_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])


ensure_package("medmnist")

try:
    ensure_package("skdim")
    import skdim  # type: ignore

    HAS_SKDIM = True
except Exception:
    HAS_SKDIM = False

import medmnist
from medmnist import INFO


# =========================
# Configuration
# =========================
@dataclass
class Config:
    dataset_flag: str = "bloodmnist"
    batch_size: int = 128
    probe_batch_size: int = 256
    learning_rate: float = 1e-3
    num_epochs: int = 20
    random_seed: int = 42


# =========================
# Model
# =========================
class SimpleMedMNISTCNN(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, input_hw: tuple[int, int] = (28, 28)) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 -> 14
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14 -> 7
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7 -> 3
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_hw)
            flat_dim = int(np.prod(self.features(dummy).shape[1:]))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# =========================
# Core utilities
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_targets(y: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    return y.view(-1).long()


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    loss_sum, acc_sum, n = 0.0, 0.0, 0

    for x, y in loader:
        x = x.to(device)
        y = prepare_targets(y).to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        acc_sum += accuracy_from_logits(logits, y)
        n += 1

    return loss_sum / max(n, 1), acc_sum / max(n, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum, acc_sum, n = 0.0, 0.0, 0

    for x, y in loader:
        x = x.to(device)
        y = prepare_targets(y).to(device)

        logits = model(x)
        loss = criterion(logits, y)

        loss_sum += loss.item()
        acc_sum += accuracy_from_logits(logits, y)
        n += 1

    return loss_sum / max(n, 1), acc_sum / max(n, 1)


@torch.no_grad()
def evaluate_model_directly(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true_all, y_pred_all, logits_all = [], [], []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_true = y_batch.view(-1).cpu().numpy()

        logits = model(x_batch)
        y_pred = logits.argmax(dim=1).cpu().numpy()

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        logits_all.append(logits.cpu().numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    logits = np.concatenate(logits_all)

    print("\nDirect model metrics")
    print("--------------------")
    print("Accuracy         :", round(accuracy_score(y_true, y_pred), 4))
    print("Balanced accuracy:", round(balanced_accuracy_score(y_true, y_pred), 4))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=90, colorbar=True)
    plt.title("Direct CNN confusion matrix")
    plt.tight_layout()
    plt.show()

    return y_true, y_pred, logits


# =========================
# Representation functions
# =========================
def extract_raw_input_matrix(loader: DataLoader, max_samples: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_x_raw, all_x_flat, all_y = [], [], []
    seen = 0

    for x_batch, y_batch in loader:
        x_raw = x_batch.detach().cpu()
        x_flat = torch.flatten(x_raw, start_dim=1)
        y = y_batch.view(-1).detach().cpu()

        all_x_raw.append(x_raw)
        all_x_flat.append(x_flat)
        all_y.append(y)

        seen += x_batch.size(0)
        if max_samples is not None and seen >= max_samples:
            break

    X_raw = torch.cat(all_x_raw, dim=0)
    X_flat = torch.cat(all_x_flat, dim=0)
    y = torch.cat(all_y, dim=0)

    if max_samples is not None:
        X_raw = X_raw[:max_samples]
        X_flat = X_flat[:max_samples]
        y = y[:max_samples]

    return X_raw.numpy(), X_flat.numpy(), y.numpy()


def run_pca(X: np.ndarray, n_components: int = 2, standardize: bool = True) -> tuple[np.ndarray, PCA, StandardScaler | None]:
    if standardize:
        scaler = StandardScaler()
        X_in = scaler.fit_transform(X)
    else:
        scaler = None
        X_in = X

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_in)
    return X_pca, pca, scaler


def plot_pca_scatter(
    X_pca: np.ndarray,
    y: np.ndarray,
    pca: PCA,
    title: str,
    class_names: list[str] | None = None,
) -> None:
    plt.figure(figsize=(12, 5))
    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=8, alpha=0.75, cmap="tab10")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var.)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var.)")
    plt.title(title)

    if class_names is not None:
        cbar = plt.colorbar(sc)
        cbar.set_ticks(np.arange(len(class_names)))
        cbar.set_ticklabels(class_names)

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def extract_representations(
    model: SimpleMedMNISTCNN,
    loader: DataLoader,
    device: torch.device,
    rep: str = "penultimate",
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_Z, all_y = [], []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch_np = y_batch.view(-1).cpu().numpy()

        if rep == "penultimate":
            h = model.features(x_batch)
            z = model.classifier[:5](h)
        elif rep == "logits":
            z = model(x_batch)
        elif rep == "conv_flat":
            h = model.features(x_batch)
            z = torch.flatten(h, start_dim=1)
        else:
            raise ValueError("rep must be one of: 'penultimate', 'logits', 'conv_flat'.")

        all_Z.append(z.detach().cpu().numpy())
        all_y.append(y_batch_np)

    return np.concatenate(all_Z, axis=0), np.concatenate(all_y, axis=0)


def summarize_representation(Z: np.ndarray, y: np.ndarray, name: str) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print("Shape:", Z.shape)

    Z_scaled = StandardScaler().fit_transform(Z)
    pca = PCA().fit(Z_scaled)

    print("PC1 variance:", round(float(pca.explained_variance_ratio_[0]), 4))
    if Z.shape[1] >= 2:
        print("PC1+PC2 variance:", round(float(pca.explained_variance_ratio_[:2].sum()), 4))

    unique, counts = np.unique(y, return_counts=True)
    if len(unique) > 1 and np.min(counts) > 1:
        print("Silhouette score:", round(float(silhouette_score(Z_scaled, y)), 4))
    else:
        print("Silhouette score: skipped")

    centroids = np.stack([Z_scaled[y == cls].mean(axis=0) for cls in unique], axis=0)
    dists = np.linalg.norm(Z_scaled[:, None, :] - centroids[None, :, :], axis=2)
    pred_labels = unique[np.argmin(dists, axis=1)]
    print("Nearest-centroid acc:", round(float(accuracy_score(y, pred_labels)), 4))


def linear_probe(train_Z: np.ndarray, train_y: np.ndarray, test_Z: np.ndarray, test_y: np.ndarray) -> LogisticRegression:
    scaler = StandardScaler()
    train_Z_scaled = scaler.fit_transform(train_Z)
    test_Z_scaled = scaler.transform(test_Z)

    clf = LogisticRegression(max_iter=2000, n_jobs=-1, multi_class="auto")
    clf.fit(train_Z_scaled, train_y)

    train_acc = accuracy_score(train_y, clf.predict(train_Z_scaled))
    test_acc = accuracy_score(test_y, clf.predict(test_Z_scaled))

    print("\nLinear probe")
    print("------------")
    print("Train accuracy:", round(float(train_acc), 4))
    print("Test accuracy :", round(float(test_acc), 4))
    return clf


def estimate_intrinsic_dimension(Z: np.ndarray, name: str, k_list: list[int] | None = None) -> dict[str, float]:
    k_list = k_list or [10, 20, 30, 50]
    print(f"\n{name}")
    print("-" * len(name))
    print("Input shape:", Z.shape)

    if not HAS_SKDIM:
        print("skdim unavailable: skipped")
        return {}

    Z_scaled = StandardScaler().fit_transform(Z)
    results: dict[str, float] = {}

    try:
        d_twonn = float(skdim.id.TwoNN().fit_transform(Z_scaled))
        results["TwoNN"] = d_twonn
        print("TwoNN:", round(d_twonn, 2))
    except Exception as e:
        print("TwoNN failed:", e)

    for k in k_list:
        print(f"\nk = {k}")
        try:
            d_mle = float(skdim.id.MLE(K=k).fit_transform(Z_scaled))
            results[f"MLE_k{k}"] = d_mle
            print("MLE :", round(d_mle, 2))
        except Exception as e:
            print("MLE failed:", e)

        try:
            d_tle = float(skdim.id.TLE().fit_transform(Z_scaled, n_neighbors=k))
            results[f"TLE_k{k}"] = d_tle
            print("TLE :", round(d_tle, 2))
        except Exception as e:
            print("TLE failed:", e)

    return results


# =========================
# Streamlined workflow
# =========================
def main() -> None:
    cfg = Config()
    set_seed(cfg.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("torch version:", torch.__version__)
    print("medmnist version:", medmnist.__version__)

    info = INFO[cfg.dataset_flag]
    n_channels = info["n_channels"]
    n_classes = len(info["label"])
    class_names = list(info["label"].values())
    DataClass = getattr(medmnist, info["python_class"])

    print("\nDataset metadata")
    print("----------------")
    print("dataset flag:", cfg.dataset_flag)
    print("task        :", info["task"])
    print("n_classes   :", n_classes)
    print("n_channels  :", n_channels)

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = DataClass(split="train", transform=transform, download=True)
    val_dataset = DataClass(split="val", transform=transform, download=True)
    test_dataset = DataClass(split="test", transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    train_loader_probe = DataLoader(train_dataset, batch_size=cfg.probe_batch_size, shuffle=False)
    test_loader_probe = DataLoader(test_dataset, batch_size=cfg.probe_batch_size, shuffle=False)

    sample_x, _ = train_dataset[0]
    _, H, W = sample_x.shape
    model = SimpleMedMNISTCNN(n_channels, n_classes, input_hw=(H, W)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None

    print("\nTraining")
    print("--------")
    for epoch in range(1, cfg.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:02d}/{cfg.num_epochs} | "
            f"train loss: {train_loss:.4f} | train acc: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f} | val acc: {val_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print("\nFinal test")
    print("----------")
    print(f"Loss: {test_loss:.4f}")
    print(f"Acc : {test_acc:.4f}")

    epochs = np.arange(1, cfg.num_epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], marker="o", label="train")
    axes[1].plot(epochs, history["val_acc"], marker="o", label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.show()

    _ = evaluate_model_directly(model, test_loader, device, class_names=class_names)

    X_raw, X_flat, y_raw = extract_raw_input_matrix(test_loader_probe)
    X_pca_raw, pca_raw, _ = run_pca(X_flat, n_components=2, standardize=True)
    plot_pca_scatter(X_pca_raw, y_raw, pca_raw, "PCA of raw input pixels", class_names)

    Z_train_pen, y_train = extract_representations(model, train_loader_probe, device, rep="penultimate")
    Z_test_pen, y_test = extract_representations(model, test_loader_probe, device, rep="penultimate")
    summarize_representation(Z_test_pen, y_test, "Penultimate representation [N, 128]")
    Z_pca_pen, pca_pen, _ = run_pca(Z_test_pen, n_components=2, standardize=True)
    plot_pca_scatter(Z_pca_pen, y_test, pca_pen, "PCA of penultimate representation", class_names)
    _ = linear_probe(Z_train_pen, y_train, Z_test_pen, y_test)

    Z_test_logits, y_test_logits = extract_representations(model, test_loader_probe, device, rep="logits")
    summarize_representation(Z_test_logits, y_test_logits, "Final logits [N, n_classes]")
    Z_pca_log, pca_log, _ = run_pca(Z_test_logits, n_components=2, standardize=True)
    plot_pca_scatter(Z_pca_log, y_test_logits, pca_log, "PCA of final logits", class_names)

    _ = estimate_intrinsic_dimension(X_flat, "Raw input pixels [N, c*h*w]")
    _ = estimate_intrinsic_dimension(Z_test_pen, "Penultimate representation [N, 128]")
    _ = estimate_intrinsic_dimension(Z_test_logits, "Final logits [N, n_classes]")


if __name__ == "__main__":
    main()