import torch
import torch.nn as nn

class ResidualConvBlock1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, alpha: float = 0.2):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.LeakyReLU(negative_slope=alpha, inplace=True)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.LeakyReLU(negative_slope=alpha, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = self.act2(x)
        return x


class UpsampleStage1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        use_conv_transpose: bool = False,
        alpha: float = 0.2,
    ):
        super().__init__()

        if use_conv_transpose:
            self.upsample = nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=2 * scale_factor,
                stride=scale_factor,
                padding=scale_factor // 2,
                output_padding=scale_factor % 2,
            )
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode="linear", align_corners=False),
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            )

        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=alpha, inplace=True)
        self.res_block = ResidualConvBlock1d(out_channels, kernel_size=3, alpha=alpha)

        if in_channels != out_channels:
            self.skip_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

        self.skip_resize = nn.Upsample(scale_factor=scale_factor, mode="linear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip_resize(self.skip_proj(x))
        x = self.upsample(x)
        x = self.bn(x)
        x = self.act(x)

        if skip.shape[-1] != x.shape[-1]:
            min_len = min(skip.shape[-1], x.shape[-1])
            skip = skip[..., :min_len]
            x = x[..., :min_len]

        x = x + skip
        x = self.res_block(x)
        return x


class DecorDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: tuple = (256, 192, 128, 96, 64, 32),
        upsample_scales: tuple = (2, 2, 2, 2, 2, 2),
        target_length: int = 24000,
        output_activation: str = "sigmoid",
        use_conv_transpose: bool = False,
        alpha: float = 0.2,
    ):
        super().__init__()

        if len(hidden_channels) != len(upsample_scales):
            raise ValueError("hidden_channels y upsample_scales deben tener la misma longitud.")

        if output_activation not in {"sigmoid", "tanh"}:
            raise ValueError("output_activation debe ser 'sigmoid' o 'tanh'.")

        self.target_length = int(target_length)

        stages = []
        current_channels = in_channels
        for next_channels, scale in zip(hidden_channels, upsample_scales):
            stages.append(
                UpsampleStage1d(
                    in_channels=current_channels,
                    out_channels=next_channels,
                    scale_factor=int(scale),
                    use_conv_transpose=use_conv_transpose,
                    alpha=alpha,
                )
            )
            current_channels = next_channels

        self.stages = nn.ModuleList(stages)
        self.out_conv = nn.Conv1d(current_channels, 1, kernel_size=3, padding=1)
        self.out_act = nn.Sigmoid() if output_activation == "sigmoid" else nn.Tanh()

    def _match_target_length(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        current_len = x.shape[-1]
        if current_len == target_length:
            return x
        if current_len > target_length:
            return x[..., :target_length]

        pad_amount = target_length - current_len
        return nn.functional.pad(x, (0, pad_amount), mode="constant", value=0.0)

    def forward(self, latents: torch.Tensor, target_length: int = None) -> torch.Tensor:
        if latents.dim() == 2:
            latents = latents.unsqueeze(-1)
        if latents.dim() != 3:
            raise ValueError("latents debe tener forma (B, C, L) o (B, C).")

        x = latents
        for stage in self.stages:
            x = stage(x)

        x = self.out_conv(x)
        x = self.out_act(x)
        final_target_length = self.target_length if target_length is None else int(target_length)
        x = self._match_target_length(x, final_target_length)
        return x


AcousticDecoder = DecorDecoder