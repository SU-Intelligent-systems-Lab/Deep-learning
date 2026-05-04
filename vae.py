"""
Variational Autoencoder (VAE) on MNIST
---------------------------------------
Key ideas:
  - Encoder maps input x to a *distribution* q(z|x) = N(mu, sigma^2)
    instead of a single point — this is what makes it "variational"
  - Reparameterization trick: z = mu + eps*sigma, eps~N(0,1)
    lets gradients flow through the sampling step
  - Loss = reconstruction loss + KL divergence
    KL term pushes q(z|x) toward the prior N(0,I), regularizing the latent space
"""

import ssl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Fix for macOS SSL certificate verification error when downloading MNIST
ssl._create_default_https_context = ssl._create_unverified_context

# ── Config ────────────────────────────────────────────────────────────────────
LATENT_DIM = 64
HIDDEN_DIM = 400
EPOCHS     = 100
BATCH_SIZE = 128
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ── Data ──────────────────────────────────────────────────────────────────────
transform = transforms.ToTensor()   # pixel values in [0, 1] — needed for BCE loss

train_loader = DataLoader(
    datasets.MNIST("data", train=True,  download=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    datasets.MNIST("data", train=False, download=True, transform=transform),
    batch_size=512, shuffle=False
)

# ── Model ─────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """Maps x → (mu, log_var) describing q(z|x)."""
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(784, HIDDEN_DIM),
            nn.ReLU()
        )
        self.fc_mu      = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc_log_var = nn.Linear(HIDDEN_DIM, LATENT_DIM)

    def forward(self, x):
        h = self.shared(x.view(-1, 784))   # flatten 28x28 → 784
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):
    """Maps z → reconstructed x, output in (0,1) via sigmoid."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 784),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, log_var):
        # During training: sample z so the decoder sees varied latent codes
        # During eval:     just return mu (deterministic)
        if self.training:
            std = torch.exp(0.5 * log_var)   # log_var → std for numerical stability
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var


# ── Loss ──────────────────────────────────────────────────────────────────────
def vae_loss(x, x_hat, mu, log_var):
    """
    ELBO loss (negated, so we minimize):
      reconstruction  — BCE between original and reconstructed pixels
      KL divergence   — closed-form KL(q(z|x) || N(0,I))
                        = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    """
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum")
    kl    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon + kl


# ── Training ──────────────────────────────────────────────────────────────────
model     = VAE().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for x, _ in train_loader:
        x = x.to(DEVICE)
        x_hat, mu, log_var = model(x)
        loss = vae_loss(x, x_hat, mu, log_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch:02d}/{EPOCHS}  loss: {avg:.2f}")


# ── Visualization 1: sample from prior and decode ─────────────────────────────
model.eval()
with torch.no_grad():
    z = torch.randn(16, LATENT_DIM).to(DEVICE)    # sample from N(0,I)
    samples = model.decoder(z).cpu()

fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i].squeeze(), cmap="gray")
    ax.axis("off")
plt.suptitle("Samples from the prior N(0, I)")
plt.tight_layout()
plt.savefig("vae_samples.png", dpi=120)
plt.show()
