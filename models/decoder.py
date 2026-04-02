import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_octave_filterbank(filterbank: nn.Conv1d, num_bands: int, fir_order: int, sample_rate: int) -> None:
    """Inicializa el banco de filtros con filtros FIR de banda de octava (ISO 266)."""
    import numpy as np
    from scipy.signal import firwin

    nyq = float(sample_rate) / 2.0
    kernel_size = int(fir_order) + 1
    # firwin requiere número impar de taps para filtros highpass/bandpass.
    design_taps = fir_order if fir_order % 2 == 1 else fir_order - 1

    fc = (
        np.array([31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0])
        if num_bands == 10
        else np.geomspace(31.5, float(sample_rate) / 4.0, num=num_bands)
    )

    weights = np.zeros((num_bands, 1, kernel_size), dtype=np.float32)
    for i, center in enumerate(fc):
        low_cut = center / np.sqrt(2)
        high_cut = center * np.sqrt(2)

        if i == 0:
            cutoff = float(np.clip(high_cut / nyq, 1e-4, 1.0 - 1e-4))
            h = firwin(design_taps, cutoff, window="hamming")
        elif i == num_bands - 1:
            cutoff = float(np.clip(low_cut / nyq, 1e-4, 1.0 - 1e-4))
            h = firwin(design_taps, cutoff, window="hamming", pass_zero=False)
        else:
            lo = float(np.clip(low_cut / nyq, 1e-4, 1.0 - 1e-4))
            hi = float(np.clip(high_cut / nyq, lo + 1e-4, 1.0 - 1e-4))
            h = firwin(design_taps, [lo, hi], window="hamming", pass_zero=False)

        pad = kernel_size - design_taps
        if pad > 0:
            h = np.concatenate([h, np.zeros(pad, dtype=np.float32)])
        weights[i, 0, :] = h.astype(np.float32)

    with torch.no_grad():
        filterbank.weight.copy_(torch.from_numpy(weights))


class DecorDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: tuple = (256, 192, 128, 96, 64, 32),
        upsample_scales: tuple = (2, 2, 2, 2, 2, 2),
        target_length: int = 45600,
        output_activation: str = "none",
        use_conv_transpose: bool = False,
        alpha: float = 0.2,
        num_bands: int = 10,
        num_decays: int = 20,
        mlp_hidden_dim: int = 512,
        mlp_hidden_layers: int = 7,
        fir_order: int = 1023,
        sample_rate: int = 48000,
        t60_min: float = 0.05,
        t60_max: float = 3.0,
        fixed_noise: bool = False,
    ):
        super().__init__()

        if output_activation not in {"sigmoid", "tanh", "none"}:
            raise ValueError("output_activation debe ser 'sigmoid', 'tanh' o 'none'.")
        if num_bands <= 0 or num_decays <= 0:
            raise ValueError("num_bands y num_decays deben ser mayores que cero.")
        if fir_order <= 0:
            raise ValueError("fir_order debe ser mayor que cero.")
        if sample_rate <= 0:
            raise ValueError("sample_rate debe ser mayor que cero.")

        self.target_length = int(target_length)
        self.num_bands = int(num_bands)
        self.num_decays = int(num_decays)
        self.sample_rate = float(sample_rate)
        self.fixed_noise = bool(fixed_noise)

        # Keep legacy args in the signature for backwards compatibility.
        self.hidden_channels = hidden_channels
        self.upsample_scales = upsample_scales
        self.use_conv_transpose = use_conv_transpose
        self.alpha = float(alpha)

        self.latent_to_hidden = nn.Linear(in_channels, mlp_hidden_dim)
        mlp_layers = []
        for _ in range(int(mlp_hidden_layers)):
            mlp_layers.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim))
            mlp_layers.append(nn.LeakyReLU(negative_slope=self.alpha, inplace=True))
        self.shared_mlp = nn.Sequential(*mlp_layers)

        head_dim = self.num_bands * self.num_decays
        self.a_head = nn.Linear(mlp_hidden_dim, head_dim)
        self.c_head = nn.Linear(mlp_hidden_dim, head_dim)

        kernel_size = int(fir_order) + 1
        self.filterbank = nn.Conv1d(
            in_channels=1,
            out_channels=self.num_bands,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        _init_octave_filterbank(self.filterbank, self.num_bands, fir_order, sample_rate)

        self.band_mixer = nn.Conv1d(self.num_bands, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.band_mixer.bias)
        if output_activation == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif output_activation == "tanh":
            self.out_act = nn.Tanh()
        else:
            self.out_act = None

        t60_values = torch.logspace(
            start=torch.log10(torch.tensor(float(t60_min))),
            end=torch.log10(torch.tensor(float(t60_max))),
            steps=self.num_decays,
        )
        # exp(-b * T60) = 1e-3  =>  b = ln(1000) / T60
        b_init = torch.log(torch.tensor(1000.0)) / t60_values
        self.decay_rates = nn.Parameter(b_init.to(torch.float32))

        self.register_buffer("_fixed_noise_buffer", torch.empty(0), persistent=False)

    def _match_target_length(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        current_len = x.shape[-1]
        if current_len == target_length:
            return x
        if current_len > target_length:
            return x[..., :target_length]

        pad_amount = target_length - current_len
        return F.pad(x, (0, pad_amount), mode="constant", value=0.0)

    def _to_latent_vector(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.dim() == 2:
            return latents
        if latents.dim() == 3:
            return latents.mean(dim=-1)
        raise ValueError("latents debe tener forma (B, C, L) o (B, C).")

    def _build_decay_basis(self, target_length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # La cola empieza a t=50 ms (después de la head), alineado con el paper (ec. 6).
        head_offset = 0.05
        n = head_offset + torch.arange(target_length, device=device, dtype=dtype) / self.sample_rate
        rates = F.softplus(self.decay_rates.to(device=device, dtype=dtype))
        basis = torch.exp(-rates.unsqueeze(-1) * n.unsqueeze(0))
        return basis

    def _sample_white_noise(self, batch_size: int, target_length: int, ref: torch.Tensor) -> torch.Tensor:
        if self.fixed_noise:
            if self._fixed_noise_buffer.numel() != target_length:
                fixed = torch.randn(1, 1, target_length, device=ref.device, dtype=ref.dtype)
                self._fixed_noise_buffer = fixed
            noise = self._fixed_noise_buffer.expand(batch_size, -1, -1)
            return noise
        return torch.randn(batch_size, 1, target_length, device=ref.device, dtype=ref.dtype)

    def _predict_amplitudes(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.latent_to_hidden(z)
        x = F.leaky_relu(x, negative_slope=self.alpha, inplace=False)
        x = self.shared_mlp(x)

        a_prime = self.a_head(x).view(-1, self.num_bands, self.num_decays)
        c_prime = self.c_head(x).view(-1, self.num_bands, self.num_decays)
        c_mask = torch.sigmoid(c_prime)
        amplitudes = a_prime * c_mask
        return amplitudes, a_prime, c_mask

    def forward(self, latents: torch.Tensor, target_length: int = None) -> torch.Tensor:
        final_target_length = self.target_length if target_length is None else int(target_length)
        z = self._to_latent_vector(latents)

        amplitudes, _, _ = self._predict_amplitudes(z)
        basis = self._build_decay_basis(
            target_length=final_target_length,
            device=z.device,
            dtype=z.dtype,
        )

        # Y = A @ E -> (B, M, T)
        envelopes = torch.matmul(amplitudes, basis)

        white_noise = self._sample_white_noise(z.size(0), final_target_length, z)
        filtered_noise = self.filterbank(white_noise)
        filtered_noise = self._match_target_length(filtered_noise, final_target_length)

        shaped_bands = envelopes * filtered_noise
        rir_tail = self.band_mixer(shaped_bands)
        if self.out_act is not None:
            rir_tail = self.out_act(rir_tail)
        return rir_tail


AcousticDecoder = DecorDecoder