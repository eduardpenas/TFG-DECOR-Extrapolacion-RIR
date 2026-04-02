import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    """
    Bloque de codificación con conexión skip:
    rama principal (Conv1D estriada) + rama residual proyectada.
    """
    def __init__(self, in_channels, out_channels, kernel_size=15, stride=2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.act = nn.PReLU()

    def forward(self, x):
        x_main = self.main(x)
        x_skip = self.skip(x)
        return self.act(x_main + x_skip)

class DecorEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        # Nueve bloques de codificación con downsampling progresivo (stride=2).
        # Alineado con el paper DECOR (Lin et al., 2025).
        self.encoder_stack = nn.Sequential(
            EncoderBlock(1, 16),      # bloque 1
            EncoderBlock(16, 32),     # bloque 2
            EncoderBlock(32, 64),     # bloque 3
            EncoderBlock(64, 128),    # bloque 4
            EncoderBlock(128, 256),   # bloque 5
            EncoderBlock(256, 512),   # bloque 6
            EncoderBlock(512, 512),   # bloque 7
            EncoderBlock(512, 512),   # bloque 8
            EncoderBlock(512, 512),   # bloque 9
        )

        # Pooling adaptativo para compactar la dimensión temporal.
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Capa lineal única para obtener el embedding z de dimensión k.
        # Alineado con el paper DECOR (Lin et al., 2025).
        self.latent_projection = nn.Linear(512, latent_dim)

    def forward(self, x):
        # x: (Batch, 1, Length)
        x = self.encoder_stack(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        z = self.latent_projection(x)  # z: (Batch, latent_dim)
        return z