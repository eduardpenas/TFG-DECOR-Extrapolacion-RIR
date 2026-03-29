import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    """
    Bloque fundamental del Encoder: Conv1D -> BatchNorm -> PReLU.
    Mantiene la estabilidad del gradiente y permite aprendizaje no lineal complejo.
    """
    def __init__(self, in_channels, out_channels, kernel_size=13, stride=2):
        super().__init__()
        # Usamos padding para mantener el control sobre la reducción temporal
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)

class DecorEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        # Arquitectura de 9 bloques según especificaciones de Aalto University
        # La dimensión temporal se reduce a la mitad en cada bloque (stride=2)
        # Esta pila es totalmente convolucional, por lo que puede operar con
        # ventanas de longitud variable.
        self.encoder_stack = nn.Sequential(
            EncoderBlock(1, 16),      # 2400 -> 1200
            EncoderBlock(16, 32),     # 1200 -> 600
            EncoderBlock(32, 64),     # 600 -> 300
            EncoderBlock(64, 128),    # 300 -> 150
            EncoderBlock(128, 256),   # 150 -> 75
            EncoderBlock(256, 512),   # 75 -> 38
            EncoderBlock(512, 512),   # 38 -> 19
            EncoderBlock(512, 512),   # 19 -> 10
            EncoderBlock(512, 512)    # 10 -> 5
        )

        # Pooling adaptativo para obtener representación temporal compacta
        # independiente de la longitud de entrada (fully convolutional).
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Proyección 1x1 al espacio latente z, manteniendo la naturaleza
        # fully convolutional de la arquitectura.
        self.latent_projection = nn.Conv1d(512, latent_dim, kernel_size=1)

    def forward(self, x):
        # x: (Batch, 1, Length)
        # Este diseño evita capas densas rígidas y permite procesar señales
        # de 50 ms, 100 ms o cualquier longitud sin fallos de dimensiones.
        x = self.encoder_stack(x)
        x = self.global_pool(x)
        z = self.latent_projection(x)
        z = z.squeeze(-1)
        # z: (Batch, latent_dim)
        return z