import torch
import torch.nn as nn
import matplotlib.pyplot as plt



# ------------------------------------------------------------# Utility function# ------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_model(model: nn.Module) -> nn.Module:
    """
    Freeze all parameters of a model.
    Useful before representation evaluation / probing.
    """
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model



@torch.no_grad()
def show_reconstructions(model, loader, device, n=8):
    model.eval()

    batch = next(iter(loader))
    x = batch["x"][:n].to(device)

    x_logits, z = model(x)
    x_recon = torch.sigmoid(x_logits)

    x = x.cpu()
    x_recon = x_recon.cpu()

    fig, axes = plt.subplots(2, n, figsize=(1.8 * n, 4))

    for i in range(n):
        axes[0, i].imshow(x[i, 0], cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=12)

        axes[1, i].imshow(x_recon[i, 0], cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstructed", fontsize=12)

    plt.tight_layout()
    plt.show()