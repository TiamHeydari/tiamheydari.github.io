import torch
import torch.nn as nn




# ------------------------------------------------------------# Autoencoder# ------------------------------------------------------------
# ------------------------------------------------------------# Autoencoder# ------------------------------------------------------------
# ------------------------------------------------------------# Autoencoder# ------------------------------------------------------------
# ------------------------------------------------------------# Autoencoder# ------------------------------------------------------------

class Autoencoder(nn.Module):
    """
    Convolutional autoencoder for dSprites.

    Input:
        x: [B, 1, 64, 64]

    Output:
        x_logits: reconstruction logits, [B, 1, 64, 64]
        z: latent representation, [B, latent_dim]

    Main methods:
        encode(x)      -> z
        decode(z)      -> x_logits
        reconstruct(x) -> sigmoid reconstruction
    """

    def __init__(self, latent_dim: int = 16):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = ConvEncoder(latent_dim=latent_dim)
        self.decoder = ConvDecoder(latent_dim=latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_logits = self.decoder(z)
        return x_logits

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_logits = self.decode(z)
        return x_logits, z

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns reconstructed images in [0, 1].
        Useful for plotting.
        """
        self.eval()
        x_logits, _ = self.forward(x)
        x_recon = torch.sigmoid(x_logits)
        return x_recon


# ------------------------------------------------------------ # ConvEncoder# ------------------------------------------------------------
class ConvEncoder(nn.Module):
    """
    Convolutional encoder for dSprites images.

    Input:
        x: [B, 1, 64, 64]

    Architecture:
        Conv2d(1,   32, kernel=4, stride=2, padding=1)  -> [B, 32, 32, 32]
        Conv2d(32,  64, kernel=4, stride=2, padding=1)  -> [B, 64, 16, 16]
        Conv2d(64, 128, kernel=4, stride=2, padding=1)  -> [B, 128, 8, 8]
        Conv2d(128,256, kernel=4, stride=2, padding=1)  -> [B, 256, 4, 4]

        Flatten -> [B, 4096]
        Linear  -> [B, 512]
        Linear  -> [B, latent_dim]

    Output:
        z: [B, latent_dim]
    """

    def __init__(self, latent_dim: int = 16):
        super().__init__()

        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        z = self.fc(h)
        return z


# ------------------------------------------------------------# ConvDecoder# ------------------------------------------------------------

class ConvDecoder(nn.Module):
    """
    Convolutional decoder for dSprites images.

    Input:
        z: [B, latent_dim]

    Architecture:
        Linear -> [B, 512]
        Linear -> [B, 4096]
        Reshape -> [B, 256, 4, 4]

        ConvTranspose2d(256,128,kernel=4,stride=2,padding=1) -> [B,128,8,8]
        ConvTranspose2d(128,64, kernel=4,stride=2,padding=1) -> [B,64,16,16]
        ConvTranspose2d(64,32,  kernel=4,stride=2,padding=1) -> [B,32,32,32]
        ConvTranspose2d(32,1,   kernel=4,stride=2,padding=1) -> [B,1,64,64]

    Output:
        x_logits: [B, 1, 64, 64]

    Note:
        The decoder returns logits, not sigmoid probabilities.
        Use BCEWithLogitsLoss during training.
        Use torch.sigmoid(x_logits) for visualization.
    """

    def __init__(self, latent_dim: int = 16):
        super().__init__()

        self.latent_dim = latent_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(z.shape[0], 256, 4, 4)
        x_logits = self.deconv(h)
        return x_logits







# ============================================================# Beta-VAE# ============================================================
# ============================================================# Beta-VAE# ============================================================
# ============================================================# Beta-VAE# ============================================================


class BetaVAE(nn.Module):
    """
    β-VAE for dSprites.

    Input:
        x: [B, 1, 64, 64]

    Encoder:
        x -> mu, logvar

    Reparameterization:
        z = mu + sigma * eps

    Decoder:
        z -> x_logits

    Important:
        encode(x) returns mu, which we use as the deterministic representation z.
    """

    def __init__(self, latent_dim: int = 16):
        super().__init__()

        self.latent_dim = latent_dim

        # -----------------------------
        # Encoder
        # -----------------------------
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # [B, 32, 32, 32]
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 16, 16]
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # [B, 128, 8, 8]
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # [B, 256, 4, 4]
            nn.ReLU(),

            nn.Flatten(),
        )

        self.encoder_out_dim = 256 * 4 * 4

        self.fc_mu = nn.Linear(self.encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_out_dim, latent_dim)

        # -----------------------------
        # Decoder
        # -----------------------------
        self.decoder_input = nn.Linear(latent_dim, self.encoder_out_dim)

        self.decoder = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(256, 4, 4)),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # [B, 128, 8, 8]
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 16, 16]
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # [B, 32, 32, 32]
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),    # [B, 1, 64, 64]
        )

    def encode_stats(self, x: torch.Tensor):
        """
        Return mu and logvar.
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        z = mu + sigma * eps
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        z = mu + eps * std

        return z

    def encode(self, x: torch.Tensor):
        """
        Deterministic representation used for probes.

        We use mu as z.
        """
        mu, logvar = self.encode_stats(x)
        return mu

    def decode(self, z: torch.Tensor):
        """
        Decode z into image logits.
        """
        h = self.decoder_input(z)
        x_logits = self.decoder(h)

        return x_logits

    def forward(self, x: torch.Tensor):
        """
        Returns:
            x_logits: reconstruction logits
            mu: latent mean
            logvar: latent log variance
            z: sampled latent
        """
        mu, logvar = self.encode_stats(x)
        z = self.reparameterize(mu, logvar)
        x_logits = self.decode(z)

        return x_logits, mu, logvar, z




# ============================================================# β-TCVAE# ============================================================
# ============================================================# β-TCVAE# ============================================================
# ============================================================# β-TCVAE# ============================================================

class BetaTCVAE(BetaVAE):
    """
    β-TCVAE for dSprites.

    Same architecture as β-VAE.

    Difference:
        β-VAE uses:
            recon + beta * KL(q(z|x) || p(z))

        β-TCVAE decomposes the KL into:
            MI + TC + dimension-wise KL

        and selectively upweights TC.

    For evaluation:
        encode(x) returns mu, same as β-VAE.
    """

    def __init__(self, latent_dim: int = 16):
        super().__init__(latent_dim=latent_dim)












