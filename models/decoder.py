import torch
import torch.nn as nn

class AcousticDecoder(nn.Module):
    def __init__(self, latent_dim=128, num_bands=10, num_decays=20):
        super().__init__()
        self.num_bands = num_bands
        self.num_decays = num_decays

        # MLP de 3 capas para procesar el vector latente
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_bands * num_decays)
        )
        
        # Máscara sigmoide para asegurar que las amplitudes sean positivas y estables
        self.activation = nn.Sigmoid()

    def forward(self, z):
        # z: (Batch, 128)
        x = self.mlp(z)
        x = self.activation(x)
        
        # Reshape a la matriz de amplitud A: (Batch, 10 bandas, 20 decaimientos)
        amplitude_matrix = x.view(-1, self.num_bands, self.num_decays)
        return amplitude_matrix