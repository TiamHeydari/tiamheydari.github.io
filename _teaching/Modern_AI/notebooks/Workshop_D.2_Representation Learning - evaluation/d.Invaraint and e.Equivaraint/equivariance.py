# equivariance.py
# ============================================================
# Equivariance analysis for learned representations
# ============================================================
#
# Goal:
#   Given paired transformation samples:
#
#       x_prime = T_p(x)
#
#   and frozen representations:
#
#       z       = F_theta(x)
#       z_prime = F_theta(x_prime)
#
#   test whether the transformed representation z_prime can be
#   predicted from the original representation z and the
#   transformation parameter p:
#
#       z_hat_prime = g(z, p)
#
#   We use a simple linear/ridge model as g.
#
# Important:
#   We also compare against the identity baseline:
#
#       z_hat_prime_identity = z
#
#   This helps distinguish true equivariance from simple invariance.
# ============================================================

from __future__ import annotations

from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def to_numpy(x):
    """
    Convert torch tensor or numpy array to numpy array.
    """
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def cosine_similarity_np(A: np.ndarray, B: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Row-wise cosine similarity between A and B.

    A, B: [N, d]
    """
    numerator = np.sum(A * B, axis=1)
    denominator = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    return numerator / (denominator + eps)


def make_equivariance_features(
    Z: np.ndarray,
    P: Optional[np.ndarray],
    feature_mode: str = "z_p",
) -> np.ndarray:
    """
    Build input features for the equivariance predictor.

    feature_mode options:
        "z_p":
            input = [z, p]

        "z_only":
            input = z

        "p_only":
            input = p

    The main setting should be "z_p".
    """
    if feature_mode not in {"z_p", "z_only", "p_only"}:
        raise ValueError("feature_mode must be one of: 'z_p', 'z_only', 'p_only'.")

    if P is None:
        P = np.zeros((Z.shape[0], 0), dtype=np.float32)

    if P.ndim == 1:
        P = P[:, None]

    if feature_mode == "z_p":
        return np.concatenate([Z, P], axis=1)

    if feature_mode == "z_only":
        return Z

    if feature_mode == "p_only":
        if P.shape[1] == 0:
            raise ValueError("P has zero columns, so feature_mode='p_only' is invalid.")
        return P

    raise ValueError("Invalid feature_mode.")


def fit_scaled_ridge(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fit ridge regression with feature and target standardization.

    Returns:
        Y_pred_test in original target scale
        fitted objects in a dictionary
    """
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    Y_train_scaled = y_scaler.fit_transform(Y_train)

    regressor = Ridge(alpha=alpha)
    regressor.fit(X_train_scaled, Y_train_scaled)

    Y_pred_scaled = regressor.predict(X_test_scaled)
    Y_pred = y_scaler.inverse_transform(Y_pred_scaled)

    fitted = {
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "regressor": regressor,
    }

    return Y_pred, fitted


def compute_equivariance_metrics(
    Z_test: np.ndarray,
    Z_prime_test: np.ndarray,
    Z_prime_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute prediction metrics for equivariance.

    Main comparison:
        model prediction:
            z_hat_prime = g(z, p)

        identity baseline:
            z_hat_prime_identity = z
    """
    Z_prime_identity = Z_test

    r2_model_uniform = r2_score(
        Z_prime_test,
        Z_prime_pred,
        multioutput="uniform_average",
    )

    r2_model_variance = r2_score(
        Z_prime_test,
        Z_prime_pred,
        multioutput="variance_weighted",
    )

    r2_identity_uniform = r2_score(
        Z_prime_test,
        Z_prime_identity,
        multioutput="uniform_average",
    )

    r2_identity_variance = r2_score(
        Z_prime_test,
        Z_prime_identity,
        multioutput="variance_weighted",
    )

    mse_model = mean_squared_error(Z_prime_test, Z_prime_pred)
    mse_identity = mean_squared_error(Z_prime_test, Z_prime_identity)

    mae_model = mean_absolute_error(Z_prime_test, Z_prime_pred)
    mae_identity = mean_absolute_error(Z_prime_test, Z_prime_identity)

    cos_model = cosine_similarity_np(Z_prime_test, Z_prime_pred)
    cos_identity = cosine_similarity_np(Z_prime_test, Z_prime_identity)

    if mse_identity > 0:
        mse_reduction_fraction = 1.0 - (mse_model / mse_identity)
    else:
        mse_reduction_fraction = np.nan

    metrics = {
        "r2_model_uniform": float(r2_model_uniform),
        "r2_model_variance": float(r2_model_variance),
        "r2_identity_uniform": float(r2_identity_uniform),
        "r2_identity_variance": float(r2_identity_variance),
        "r2_gain_uniform": float(r2_model_uniform - r2_identity_uniform),
        "r2_gain_variance": float(r2_model_variance - r2_identity_variance),
        "mse_model": float(mse_model),
        "mse_identity": float(mse_identity),
        "mse_reduction_fraction": float(mse_reduction_fraction),
        "mae_model": float(mae_model),
        "mae_identity": float(mae_identity),
        "mean_cosine_model": float(np.mean(cos_model)),
        "std_cosine_model": float(np.std(cos_model)),
        "mean_cosine_identity": float(np.mean(cos_identity)),
        "std_cosine_identity": float(np.std(cos_identity)),
    }

    return metrics


# ------------------------------------------------------------
# Core equivariance fitting
# ------------------------------------------------------------

def fit_linear_equivariance_model(
    Z: np.ndarray,
    Z_prime: np.ndarray,
    P: Optional[np.ndarray],
    target_mode: str = "delta",
    feature_mode: str = "z_p",
    alpha: float = 1.0,
    test_size: float = 0.25,
    seed: int = 0,
    return_predictions: bool = False,
) -> Dict[str, Any]:
    """
    Fit a linear/ridge equivariance predictor.

    Inputs:
        Z:
            original representation [N, d]

        Z_prime:
            transformed representation [N, d]

        P:
            transformation parameter [N, p_dim]

    target_mode:
        "direct":
            learn z_prime = g(z, p)

        "delta":
            learn delta_z = z_prime - z
            then predict z_hat_prime = z + delta_hat

            This is recommended because it explicitly compares against
            the identity/invariance baseline.

    feature_mode:
        "z_p":
            g receives [z, p]

        "z_only":
            g receives z only

        "p_only":
            g receives p only

    Returns:
        result dictionary with metrics and optional predictions.
    """
    if target_mode not in {"direct", "delta"}:
        raise ValueError("target_mode must be one of: 'direct', 'delta'.")

    Z = to_numpy(Z).astype(np.float32)
    Z_prime = to_numpy(Z_prime).astype(np.float32)

    if P is None:
        P = np.zeros((Z.shape[0], 0), dtype=np.float32)
    else:
        P = to_numpy(P).astype(np.float32)

    if P.ndim == 1:
        P = P[:, None]

    if Z.shape != Z_prime.shape:
        raise ValueError(
            f"Z and Z_prime must have the same shape. Got {Z.shape} and {Z_prime.shape}."
        )

    if Z.shape[0] != P.shape[0]:
        raise ValueError(
            f"Z and P must have the same number of samples. Got {Z.shape[0]} and {P.shape[0]}."
        )

    X = make_equivariance_features(
        Z=Z,
        P=P,
        feature_mode=feature_mode,
    )

    if target_mode == "direct":
        Y = Z_prime
    else:
        Y = Z_prime - Z

    indices = np.arange(Z.shape[0])

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    X_train = X[train_idx]
    X_test = X[test_idx]

    Y_train = Y[train_idx]

    Z_test = Z[test_idx]
    Z_prime_test = Z_prime[test_idx]

    Y_pred_test, fitted = fit_scaled_ridge(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        alpha=alpha,
    )

    if target_mode == "direct":
        Z_prime_pred = Y_pred_test
    else:
        Z_prime_pred = Z_test + Y_pred_test

    metrics = compute_equivariance_metrics(
        Z_test=Z_test,
        Z_prime_test=Z_prime_test,
        Z_prime_pred=Z_prime_pred,
    )

    result = {
        **metrics,
        "target_mode": target_mode,
        "feature_mode": feature_mode,
        "alpha": float(alpha),
        "test_size": float(test_size),
        "seed": int(seed),
        "n_samples": int(Z.shape[0]),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "latent_dim": int(Z.shape[1]),
        "p_dim": int(P.shape[1]),
        "feature_dim": int(X.shape[1]),
    }

    if return_predictions:
        result.update(
            {
                "Z_test": Z_test,
                "Z_prime_test": Z_prime_test,
                "Z_prime_pred": Z_prime_pred,
                "P_test": P[test_idx],
                "train_idx": train_idx,
                "test_idx": test_idx,
                "fitted": fitted,
            }
        )

    return result


# ------------------------------------------------------------
# Run from encoded invariance results
# ------------------------------------------------------------

def run_equivariance_analysis_from_encoded(
    invariance_results: Dict[str, Dict[str, Dict[str, Any]]],
    target_mode: str = "delta",
    feature_mode: str = "z_p",
    alpha: float = 1.0,
    test_size: float = 0.25,
    seed: int = 0,
    return_predictions: bool = False,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Dict[str, Any]]]]:
    """
    Run equivariance analysis using encoded arrays already stored
    inside invariance_results.

    This expects each result to contain:
        "Z"
        "Z_prime"
        "P"

    This is available if invariance was run with:
        return_encoded=True
    """
    records = []
    equivariance_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for model_name, transform_dict in invariance_results.items():
        if verbose:
            print(f"\nEvaluating model: {model_name}")

        equivariance_results[model_name] = {}

        for transform_name, result in transform_dict.items():
            missing = [key for key in ["Z", "Z_prime", "P"] if key not in result]

            if len(missing) > 0:
                raise KeyError(
                    f"Missing {missing} for {model_name}/{transform_name}. "
                    "Run invariance analysis with return_encoded=True, "
                    "or use run_equivariance_analysis_from_loaders()."
                )

            eq_result = fit_linear_equivariance_model(
                Z=result["Z"],
                Z_prime=result["Z_prime"],
                P=result["P"],
                target_mode=target_mode,
                feature_mode=feature_mode,
                alpha=alpha,
                test_size=test_size,
                seed=seed,
                return_predictions=return_predictions,
            )

            equivariance_results[model_name][transform_name] = eq_result

            record = {
                "model": model_name,
                "transformation": transform_name,
                "target_mode": target_mode,
                "feature_mode": feature_mode,
                "alpha": alpha,
                "n_samples": eq_result["n_samples"],
                "n_train": eq_result["n_train"],
                "n_test": eq_result["n_test"],
                "latent_dim": eq_result["latent_dim"],
                "p_dim": eq_result["p_dim"],
                "feature_dim": eq_result["feature_dim"],
                "r2_model_uniform": eq_result["r2_model_uniform"],
                "r2_model_variance": eq_result["r2_model_variance"],
                "r2_identity_uniform": eq_result["r2_identity_uniform"],
                "r2_identity_variance": eq_result["r2_identity_variance"],
                "r2_gain_uniform": eq_result["r2_gain_uniform"],
                "r2_gain_variance": eq_result["r2_gain_variance"],
                "mse_model": eq_result["mse_model"],
                "mse_identity": eq_result["mse_identity"],
                "mse_reduction_fraction": eq_result["mse_reduction_fraction"],
                "mae_model": eq_result["mae_model"],
                "mae_identity": eq_result["mae_identity"],
                "mean_cosine_model": eq_result["mean_cosine_model"],
                "mean_cosine_identity": eq_result["mean_cosine_identity"],
            }

            records.append(record)

            if verbose:
                print(
                    f"  {transform_name:15s} | "
                    f"R2(model) = {eq_result['r2_model_variance']:.4f} | "
                    f"R2(identity) = {eq_result['r2_identity_variance']:.4f} | "
                    f"gain = {eq_result['r2_gain_variance']:.4f}"
                )

    equivariance_df = pd.DataFrame(records)

    return equivariance_df, equivariance_results


def run_equivariance_analysis_from_loaders(
    models: Dict[str, torch.nn.Module],
    pair_loaders: Dict[str, Any],
    device: Optional[torch.device] = None,
    target_mode: str = "delta",
    feature_mode: str = "z_p",
    alpha: float = 1.0,
    test_size: float = 0.25,
    seed: int = 0,
    return_predictions: bool = False,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Dict[str, Any]]]]:
    """
    Run equivariance analysis directly from models and paired loaders.

    This re-encodes the paired data using invariance.encode_loader_pairs.
    """
    from invariance import encode_loader_pairs, get_default_device

    if device is None:
        device = get_default_device()

    encoded_results = {}

    for model_name, model in models.items():
        encoded_results[model_name] = {}

        for transform_name, loader in pair_loaders.items():
            encoded = encode_loader_pairs(
                model=model,
                pair_loader=loader,
                device=device,
            )

            encoded_results[model_name][transform_name] = {
                "Z": encoded["Z"],
                "Z_prime": encoded["Z_prime"],
                "P": encoded["P"],
            }

    return run_equivariance_analysis_from_encoded(
        invariance_results=encoded_results,
        target_mode=target_mode,
        feature_mode=feature_mode,
        alpha=alpha,
        test_size=test_size,
        seed=seed,
        return_predictions=return_predictions,
        verbose=verbose,
    )


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_equivariance_bar(
    equivariance_df: pd.DataFrame,
    value_col: str = "r2_model_variance",
    title: str = "Equivariance score",
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (9, 4),
):
    """
    Bar plot of equivariance score.

    Recommended value_col:
        "r2_model_variance"
    """
    pivot = equivariance_df.pivot(
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

    ax.axhline(0.0, linestyle="--", linewidth=1)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

    return pivot


def plot_equivariance_gain_bar(
    equivariance_df: pd.DataFrame,
    value_col: str = "r2_gain_variance",
    title: str = "Equivariance gain over identity baseline",
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (9, 4),
):
    """
    Plot how much better g(z,p) predicts z_prime compared with
    the identity baseline z_prime_hat = z.

    This is often the most important equivariance plot.
    """
    pivot = equivariance_df.pivot(
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

    ax.axhline(0.0, linestyle="--", linewidth=1)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

    return pivot


def plot_identity_vs_model_r2(
    equivariance_df: pd.DataFrame,
    model_name: str,
    r2_model_col: str = "r2_model_variance",
    r2_identity_col: str = "r2_identity_variance",
    figsize: Tuple[float, float] = (8, 4),
):
    """
    For one representation model, compare:
        R2 of learned equivariance predictor
        R2 of identity baseline
    """
    sub = equivariance_df[equivariance_df["model"] == model_name].copy()

    if len(sub) == 0:
        raise ValueError(f"No rows found for model_name={model_name}.")

    sub = sub.set_index("transformation")[
        [r2_model_col, r2_identity_col]
    ]

    ax = sub.plot(
        kind="bar",
        figsize=figsize,
    )

    ax.set_title(f"Equivariance predictor vs identity baseline: {model_name}")
    ax.set_xlabel("Transformation T")
    ax.set_ylabel("R2")
    ax.axhline(0.0, linestyle="--", linewidth=1)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

    return sub


def plot_equivariance_heatmap(
    equivariance_df: pd.DataFrame,
    value_col: str = "r2_model_variance",
    title: str = "Equivariance heatmap",
    figsize: Tuple[float, float] = (6, 4),
):
    """
    Simple heatmap of equivariance scores.
    """
    pivot = equivariance_df.pivot(
        index="transformation",
        columns="model",
        values=value_col,
    )

    values = pivot.values

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(values, aspect="auto")

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


def plot_predicted_vs_true_dimension(
    equivariance_results: Dict[str, Dict[str, Dict[str, Any]]],
    model_name: str,
    transform_name: str,
    dim: int = 0,
    figsize: Tuple[float, float] = (4, 4),
):
    """
    Scatter plot for one latent dimension:

        true z'_dim vs predicted z'_dim

    Requires return_predictions=True.
    """
    result = equivariance_results[model_name][transform_name]

    required = ["Z_prime_test", "Z_prime_pred"]
    missing = [key for key in required if key not in result]

    if len(missing) > 0:
        raise KeyError(
            f"Missing {missing}. Run with return_predictions=True."
        )

    true_vals = result["Z_prime_test"][:, dim]
    pred_vals = result["Z_prime_pred"][:, dim]

    plt.figure(figsize=figsize)
    plt.scatter(true_vals, pred_vals, s=8, alpha=0.4)

    min_val = min(true_vals.min(), pred_vals.min())
    max_val = max(true_vals.max(), pred_vals.max())

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1)

    plt.xlabel(r"true $z'_m$")
    plt.ylabel(r"predicted $\hat{z}'_m$")
    plt.title(f"{model_name} | {transform_name} | dim {dim}")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Interpretation helper
# ------------------------------------------------------------

def print_equivariance_interpretation():
    """
    Print short interpretation guide.
    """
    print("Interpretation:")
    print("  Equivariance asks whether the transformed representation z' can")
    print("  be predicted from the original representation z and transformation p.")
    print()
    print("  We fit:")
    print("      z_hat' = z + g(z,p)")
    print()
    print("  R2(model) high:")
    print("      z' is predictable from z and p.")
    print()
    print("  R2(identity) high:")
    print("      z' is already close to z; this may reflect invariance.")
    print()
    print("  Gain over identity > 0:")
    print("      the transformation parameter helps predict how z changes.")
    print()
    print("Key distinction:")
    print("  Invariance:   z' ≈ z")
    print("  Equivariance: z' changes, but in a predictable way.")


def run_and_plot_equivariance_from_encoded(
    invariance_results: Dict[str, Dict[str, Dict[str, Any]]],
    target_mode: str = "delta",
    feature_mode: str = "z_p",
    alpha: float = 1.0,
    test_size: float = 0.25,
    seed: int = 0,
    return_predictions: bool = False,
    verbose: bool = True,
):
    """
    Convenience wrapper:
        1. run equivariance analysis from encoded results
        2. plot R2(model)
        3. plot gain over identity
        4. print interpretation
    """
    equivariance_df, equivariance_results = run_equivariance_analysis_from_encoded(
        invariance_results=invariance_results,
        target_mode=target_mode,
        feature_mode=feature_mode,
        alpha=alpha,
        test_size=test_size,
        seed=seed,
        return_predictions=return_predictions,
        verbose=verbose,
    )

    plot_equivariance_bar(
        equivariance_df,
        value_col="r2_model_variance",
        title="Equivariance score: predicting transformed representation",
    )

    plot_equivariance_gain_bar(
        equivariance_df,
        value_col="r2_gain_variance",
        title="Equivariance gain over identity baseline",
    )

    print_equivariance_interpretation()

    return equivariance_df, equivariance_results