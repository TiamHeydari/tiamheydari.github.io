# data.py

from pathlib import Path

import requests
import numpy as np
import torch

from tqdm.auto import tqdm
from torch.utils.data import Dataset


# ------------------------------------------------------------
# dSprites download settings
# ------------------------------------------------------------

DSPRITES_URL = (
    "https://github.com/google-deepmind/dsprites-dataset/raw/refs/heads/master/"
    "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
)

DEFAULT_DATA_DIR = Path("./data/dsprites")
DEFAULT_DSPRITES_FILENAME = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

FACTOR_NAMES = [
    "color",
    "shape",
    "scale",
    "rotation",
    "x_position",
    "y_position",
]

FACTOR_INDEX = {name: i for i, name in enumerate(FACTOR_NAMES)}


# ------------------------------------------------------------
# Download utilities
# ------------------------------------------------------------

def download_file(url: str, path: str | Path, desc: str = "Downloading") -> Path:
    """
    Download a file with a progress bar if it does not already exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        print(f"File already exists: {path}")
        return path

    print(f"Downloading to: {path}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(path, "wb") as f, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=desc,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    return path


def download_dsprites(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    filename: str = DEFAULT_DSPRITES_FILENAME,
    url: str = DSPRITES_URL,
) -> Path:
    """
    Download the dSprites dataset if needed.

    Returns
    -------
    Path
        Local path to the dSprites .npz file.
    """
    data_dir = Path(data_dir)
    dsprites_path = data_dir / filename

    return download_file(
        url=url,
        path=dsprites_path,
        desc="Downloading dSprites",
    )


# ------------------------------------------------------------
# PyTorch Dataset wrapper
# ------------------------------------------------------------

class DSpritesDataset(Dataset):
    """
    Clean PyTorch dataset for dSprites.

    Each item returns:
        x: image tensor, shape [1, 64, 64]
        y: discrete factor class vector, shape [6]
        y_values: continuous/original factor value vector, shape [6]

    Factor order:
        0 = color
        1 = shape
        2 = scale
        3 = rotation
        4 = x_position
        5 = y_position
    """

    def __init__(
        self,
        X: torch.Tensor,
        Y_classes: torch.Tensor,
        Y_values: torch.Tensor,
        factor_names: list[str] = FACTOR_NAMES,
    ):
        self.X = X
        self.Y_classes = Y_classes
        self.Y_values = Y_values
        self.factor_names = factor_names
        self.factor_index = {name: i for i, name in enumerate(factor_names)}

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.X[idx],                    # [1, 64, 64]
            "y": self.Y_classes[idx],            # [6]
            "y_values": self.Y_values[idx],      # [6]
        }

    def get_factor(
        self,
        idx: int,
        factor_name: str,
        values: bool = False,
    ) -> torch.Tensor:
        """
        Get one factor for one sample.

        Parameters
        ----------
        idx:
            Sample index.

        factor_name:
            One of:
                "color", "shape", "scale",
                "rotation", "x_position", "y_position"

        values:
            If False, return discrete class label.
            If True, return continuous/original factor value.
        """
        factor_idx = self.factor_index[factor_name]

        if values:
            return self.Y_values[idx, factor_idx]

        return self.Y_classes[idx, factor_idx]

    def get_all_factor(
        self,
        factor_name: str,
        values: bool = False,
    ) -> torch.Tensor:
        """
        Get one factor for all samples.

        Example
        -------
        all_shapes = dataset.get_all_factor("shape")
        """
        factor_idx = self.factor_index[factor_name]

        if values:
            return self.Y_values[:, factor_idx]

        return self.Y_classes[:, factor_idx]


# ------------------------------------------------------------
# Load dSprites
# ------------------------------------------------------------

def load_dSprites(
    dsprites_path: str | Path,
    factor_names: list[str] = FACTOR_NAMES,
    verbose: bool = True,
) -> dict:
    """
    Load dSprites from a local .npz file and return tensors plus a Dataset.

    Parameters
    ----------
    dsprites_path:
        Path to the downloaded dSprites .npz file.

    factor_names:
        Names assigned to the six dSprites factors.

    verbose:
        If True, print dataset information.

    Returns
    -------
    dict
        Dictionary containing:
            dataset
            X
            Y_classes
            Y_values
            metadata
            factor_names
            factor_index
            imgs
            latents_classes
            latents_values
    """
    dsprites_path = Path(dsprites_path)

    if not dsprites_path.exists():
        raise FileNotFoundError(
            f"Could not find dSprites file at: {dsprites_path}\n"
            "Run download_dsprites() first."
        )

    # dSprites metadata was saved with older Python pickle encoding.
    # Without encoding='latin1', loading metadata can fail in Python 3.
    dataset_npz = np.load(
        dsprites_path,
        allow_pickle=True,
        encoding="latin1",
    )

    imgs = dataset_npz["imgs"]                              # [N, 64, 64]
    latents_classes = dataset_npz["latents_classes"]        # [N, 6]
    latents_values = dataset_npz["latents_values"]          # [N, 6]
    metadata = dataset_npz["metadata"][()]

    # Convert raw arrays into clean tensors.
    X = torch.tensor(imgs, dtype=torch.float32).unsqueeze(1)         # [N, 1, 64, 64]
    Y_classes = torch.tensor(latents_classes, dtype=torch.long)      # [N, 6]
    Y_values = torch.tensor(latents_values, dtype=torch.float32)     # [N, 6]

    dataset = DSpritesDataset(
        X=X,
        Y_classes=Y_classes,
        Y_values=Y_values,
        factor_names=factor_names,
    )

    factor_index = {name: i for i, name in enumerate(factor_names)}

    if verbose:
        print("Keys in dataset:")
        print(dataset_npz.files)

        print("\nShapes:")
        print("imgs:", imgs.shape)
        print("latents_classes:", latents_classes.shape)
        print("latents_values:", latents_values.shape)

        print("\nLatent names from metadata:")
        print(metadata["latents_names"])

        print("\nLatent sizes from metadata:")
        print(metadata["latents_sizes"])

        print("\nTensor shapes:")
        print("X:", X.shape)
        print("Y_classes:", Y_classes.shape)
        print("Y_values:", Y_values.shape)

        print("\nFactor index:")
        for name, idx in factor_index.items():
            unique_vals = torch.unique(Y_classes[:, idx])
            print(
                f"{idx}: {name:12s} | "
                f"{len(unique_vals):2d} unique class values | "
                f"{unique_vals.tolist()}"
            )

        print("\nDataset length:", len(dataset))

        sample = dataset[0]
        print("\nOne sample:")
        print("x:", sample["x"].shape)
        print("y:", sample["y"])
        print("y_values:", sample["y_values"])

        print("\nNote:")
        print("Color is included but constant in original dSprites.")

    return {
        "dataset": dataset,
        "X": X,
        "Y_classes": Y_classes,
        "Y_values": Y_values,
        "metadata": metadata,
        "factor_names": factor_names,
        "factor_index": factor_index,
        "imgs": imgs,
        "latents_classes": latents_classes,
        "latents_values": latents_values,
    }


# Cleaner Python-style alias.
def load_dsprites(
    dsprites_path: str | Path,
    factor_names: list[str] = FACTOR_NAMES,
    verbose: bool = True,
) -> dict:
    """
    Alias for load_dSprites().
    """
    return load_dSprites(
        dsprites_path=dsprites_path,
        factor_names=factor_names,
        verbose=verbose,
    )