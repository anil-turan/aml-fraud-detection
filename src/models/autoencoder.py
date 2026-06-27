"""
LSTM Autoencoder for sequence-aware transaction anomaly detection.

Architecture:
  Input (seq_len, n_features)
    → LSTM Encoder  (hidden_dim → latent_dim)
    → repeat vector
    → LSTM Decoder  (latent_dim → hidden_dim)
    → Linear output (reconstruct input)

Anomaly score = mean squared reconstruction error per sample.
High reconstruction error → the model struggles to compress the pattern → anomalous.

For tabular (non-sequential) data we reshape each transaction as a
sequence of length 1, letting the LSTM learn temporal context when
windowed sequences are provided (see create_sequences()).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.enc_to_latent = nn.Linear(hidden_dim, latent_dim)

        self.latent_to_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, n_features)
        enc_out, _ = self.encoder(x)
        latent = self.enc_to_latent(enc_out[:, -1, :])  # last hidden state

        dec_input = self.latent_to_dec(latent).unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(dec_input)
        reconstructed = self.output_layer(dec_out)
        return reconstructed, latent


def create_sequences(X: np.ndarray, seq_len: int = 1) -> np.ndarray:
    """Reshape (n_samples, n_features) → (n_samples, seq_len, n_features)."""
    if seq_len == 1:
        return X[:, np.newaxis, :]
    seqs = []
    for i in range(len(X) - seq_len + 1):
        seqs.append(X[i : i + seq_len])
    return np.stack(seqs)


def train_autoencoder(
    X_train: np.ndarray,
    n_features: int,
    hidden_dim: int = 64,
    latent_dim: int = 16,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    seq_len: int = 1,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[LSTMAutoencoder, list]:
    X_seq = create_sequences(X_train, seq_len)
    tensor = torch.FloatTensor(X_seq).to(device)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)

    model = LSTMAutoencoder(n_features, hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.MSELoss()

    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        scheduler.step()
        avg_loss = epoch_loss / len(tensor)
        history.append(avg_loss)
        if verbose and epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.6f}")

    return model, history


def reconstruction_errors(
    model: LSTMAutoencoder,
    X: np.ndarray,
    seq_len: int = 1,
    batch_size: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    model.eval()
    X_seq = create_sequences(X, seq_len)
    tensor = torch.FloatTensor(X_seq).to(device)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)

    errors = []
    with torch.no_grad():
        for (batch,) in loader:
            recon, _ = model(batch)
            mse = ((recon - batch) ** 2).mean(dim=(1, 2))
            errors.append(mse.cpu().numpy())

    return np.concatenate(errors)


def save_model(model: LSTMAutoencoder, path: str | Path) -> None:
    torch.save(model.state_dict(), path)


def load_model(path: str | Path, n_features: int, **kwargs) -> LSTMAutoencoder:
    model = LSTMAutoencoder(n_features, **kwargs)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
