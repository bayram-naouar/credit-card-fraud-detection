import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.config import MODELS, PREPROCESSING
from src.models.base import BaseModel


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoEncoderModel(BaseModel):
    def __init__(self, input_dim, **kwargs):
        super().__init__()
        base_params = MODELS["autoencoder"].copy()
        torch.manual_seed(PREPROCESSING["random_state"])
        base_params.update(kwargs)
        self.model = AutoEncoder(input_dim, int(base_params["latent_dim"]))
        self.model_name = "autoencoder"
        self.threshold = None
        self.input_dim = input_dim
        self.params = base_params.copy()

    def anomaly_scores(self, X):
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
        errors = ((X_tensor - reconstructed) ** 2).mean(dim=1)
        return errors.numpy()

    def _fit(self, X_train):
        X_tensor = torch.FloatTensor(X_train)
        dataset = TensorDataset(X_tensor, X_tensor)
        loader = DataLoader(dataset, batch_size=self.params["batch_size"], shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.params["learning_rate"])
        criterion = nn.MSELoss()
        for epoch in range(self.params["epochs"]):
            epoch_loss = 0
            for batch_X, _ in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_X)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(
                f"[autoencoder] Epoch {epoch + 1}/{self.params['epochs']} | Loss: {epoch_loss / len(loader):.6f}"
            )
