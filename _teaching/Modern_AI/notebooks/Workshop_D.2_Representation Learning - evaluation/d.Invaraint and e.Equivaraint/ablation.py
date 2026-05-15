# ============================================================
# ablation.py
#
# c-Disentanglement: c3. Ablation
#
# We remove one latent dimension at a time:
#
#     z -> z_{-m}
#
# and measure how much factor prediction drops:
#
#     A_{k,m} = Score_k(z) - Score_k(z_{-m})
# ============================================================


from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# ============================================================
# Basic utilities
# ============================================================

def get_device() -> torch.device:
    """
    Use GPU if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalized_accuracy(
    acc: float,
    num_classes: int,
) -> float:
    """
    Chance-corrected accuracy.

        NormAcc = (Acc - Chance) / (1 - Chance)
    """
    chance = 1.0 / num_classes

    return (acc - chance) / (1.0 - chance)


@torch.no_grad()
def evaluate_probe_on_Z(
    probe_model,
    Z: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 512,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Evaluate a trained probe on a representation matrix Z.

    Args:
        probe_model:
            trained factor probe

        Z:
            representation matrix [N, D_z]

        y:
            factor labels [N]

    Returns:
        dictionary with accuracy and loss
    """
    if device is None:
        device = get_device()

    probe_model = probe_model.to(device)
    probe_model.eval()

    Z = Z.detach().float()
    y = y.detach().long()

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    total_correct = 0
    total_loss = 0.0
    total_samples = 0

    for start in range(0, Z.shape[0], batch_size):
        end = start + batch_size

        Z_batch = Z[start:end].to(device)
        y_batch = y[start:end].to(device)

        logits = probe_model(Z_batch)

        loss = criterion(logits, y_batch)

        pred = logits.argmax(dim=1)

        total_correct += (pred == y_batch).sum().item()
        total_loss += loss.item()
        total_samples += y_batch.shape[0]

    acc = total_correct / max(total_samples, 1)
    loss = total_loss / max(total_samples, 1)

    return {
        "acc": acc,
        "loss": loss,
    }


# ============================================================
# Ablation
# ============================================================

def ablate_one_dimension(
    Z: torch.Tensor,
    dim: int,
    replacement_value: float,
) -> torch.Tensor:
    """
    Remove one latent dimension by replacing it with a reference value.

    We do not literally delete the dimension because the trained probe
    expects the same input size.

    Instead:

        z_{i,m} <- replacement_value

    Usually:

        replacement_value = mean of z_m on the training set
    """
    Z_ablated = Z.detach().clone()
    Z_ablated[:, dim] = replacement_value

    return Z_ablated


def run_latent_dimension_ablation(
    Z_train: torch.Tensor,
    Z_test: torch.Tensor,
    Y_test: torch.Tensor,
    probe_models: Dict,
    factor_names: Sequence[str],
    skip_factors=("color",),
    batch_size: int = 512,
    device: torch.device = None,
    score_type: str = "normalized_acc",
    verbose: bool = True,
) -> Tuple[np.ndarray, list, pd.DataFrame]:
    """
    Run single-latent-dimension ablation.

    For each factor k and latent dimension m:

        A_{k,m} = Score_k(z) - Score_k(z_{-m})

    where z_{-m} is produced by replacing dimension m
    with the training-set mean of that dimension.

    Args:
        Z_train:
            training representations [N_train, D_z]

        Z_test:
            test representations [N_test, D_z]

        Y_test:
            test factor labels [N_test, num_factors]

        probe_models:
            dictionary returned by run_all_factor_probes

        factor_names:
            list of factor names

        skip_factors:
            factors to skip, usually ("color",)

        score_type:
            "acc" or "normalized_acc"

    Returns:
        A:
            ablation matrix [num_factors, D_z]

        used_factors:
            factors included in the analysis

        records:
            long-form dataframe with all results
    """
    if device is None:
        device = get_device()

    factor_names = [str(f) for f in factor_names]

    Z_train = Z_train.detach().float().cpu()
    Z_test = Z_test.detach().float().cpu()
    Y_test = Y_test.detach().long().cpu()

    latent_dim = Z_test.shape[1]

    # Mean replacement keeps ablated points near the latent distribution.
    z_train_mean = Z_train.mean(dim=0)

    used_factors = [
        f for f in factor_names
        if f not in skip_factors
    ]

    A = np.zeros((len(used_factors), latent_dim), dtype=float)

    records = []

    for factor_row, factor_name in enumerate(used_factors):
        factor_idx = factor_names.index(factor_name)

        y_factor = Y_test[:, factor_idx]

        num_classes = int(torch.unique(y_factor).numel())

        probe_model = probe_models[factor_name]

        # ----------------------------------------------------
        # Baseline score using full z
        # ----------------------------------------------------
        baseline_metrics = evaluate_probe_on_Z(
            probe_model=probe_model,
            Z=Z_test,
            y=y_factor,
            batch_size=batch_size,
            device=device,
        )

        baseline_acc = baseline_metrics["acc"]

        if score_type == "acc":
            baseline_score = baseline_acc

        elif score_type == "normalized_acc":
            baseline_score = normalized_accuracy(
                acc=baseline_acc,
                num_classes=num_classes,
            )

        else:
            raise ValueError("score_type must be 'acc' or 'normalized_acc'")

        if verbose:
            print(
                f"{factor_name:12s} | "
                f"baseline {score_type}: {baseline_score:.3f}"
            )

        # ----------------------------------------------------
        # Ablate each latent dimension
        # ----------------------------------------------------
        for dim in range(latent_dim):
            Z_ablated = ablate_one_dimension(
                Z=Z_test,
                dim=dim,
                replacement_value=z_train_mean[dim].item(),
            )

            ablated_metrics = evaluate_probe_on_Z(
                probe_model=probe_model,
                Z=Z_ablated,
                y=y_factor,
                batch_size=batch_size,
                device=device,
            )

            ablated_acc = ablated_metrics["acc"]

            if score_type == "acc":
                ablated_score = ablated_acc

            else:
                ablated_score = normalized_accuracy(
                    acc=ablated_acc,
                    num_classes=num_classes,
                )

            drop = baseline_score - ablated_score

            A[factor_row, dim] = drop

            records.append(
                {
                    "factor": factor_name,
                    "dim": dim,
                    "num_classes": num_classes,
                    "baseline_acc": baseline_acc,
                    "ablated_acc": ablated_acc,
                    "baseline_score": baseline_score,
                    "ablated_score": ablated_score,
                    "drop": drop,
                    "score_type": score_type,
                }
            )

    records = pd.DataFrame(records)

    return A, used_factors, records


# ============================================================
# Plotting
# ============================================================

def plot_ablation_heatmap(
    A,
    factor_names,
    title: str = "Latent dimension ablation",
    score_label: str = "Score drop after ablation",
    figsize=None,
):
    """
    Plot ablation heatmap.

    Rows:
        factors

    Columns:
        ablated latent dimensions

    Values:
        A_{k,m} = Score_k(z) - Score_k(z_{-m})
    """
    A = np.asarray(A)

    num_factors, latent_dim = A.shape

    if figsize is None:
        figsize = (0.6 * latent_dim + 3, 0.5 * num_factors + 2.5)

    plt.figure(figsize=figsize)

    im = plt.imshow(A, aspect="auto")

    plt.colorbar(im, label=score_label)

    plt.yticks(
        ticks=np.arange(num_factors),
        labels=factor_names,
    )

    plt.xticks(
        ticks=np.arange(latent_dim),
        labels=[f"z{m}" for m in range(latent_dim)],
        rotation=90,
    )

    plt.xlabel("Ablated latent dimension")
    plt.ylabel("Predicted factor")
    plt.title(title)

    plt.tight_layout()
    plt.show()


def get_top_ablation_dimensions(
    ablation_records: pd.DataFrame,
    top_k: int = 3,
) -> pd.DataFrame:
    """
    Return top-k most important latent dimensions per factor.

    Importance is measured by score drop.
    """
    top = (
        ablation_records
        .sort_values(["factor", "drop"], ascending=[True, False])
        .groupby("factor")
        .head(top_k)
        .reset_index(drop=True)
    )

    return top