from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_edc_shape(x: torch.Tensor, name: str) -> None:
	if x.dim() != 3:
		raise ValueError(f"{name} debe tener forma (Batch, 1, Length).")
	if x.size(1) != 1:
		raise ValueError(f"{name} debe tener un único canal (dim=1 igual a 1).")


class MultiResolutionSTFTLoss(nn.Module):
	def __init__(
		self,
		fft_sizes: Sequence[int] = (64, 512, 2048, 8192),
		hop_sizes: Sequence[int] = (32, 256, 1024, 4096),
		win_lengths: Sequence[int] = (64, 512, 2048, 8192),
		eps: float = 1e-7,
	):
		super().__init__()

		if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths)):
			raise ValueError("fft_sizes, hop_sizes y win_lengths deben tener la misma longitud.")

		self.resolutions: Tuple[Tuple[int, int, int], ...] = tuple(
			(int(n_fft), int(hop), int(win_len))
			for n_fft, hop, win_len in zip(fft_sizes, hop_sizes, win_lengths)
		)
		self.eps = float(eps)

	def _stft_magnitude(
		self,
		signal: torch.Tensor,
		n_fft: int,
		hop_length: int,
		win_length: int,
	) -> torch.Tensor:
		length = signal.size(-1)
		n_fft_eff = max(2, min(n_fft, length))
		hop_eff = max(1, min(hop_length, n_fft_eff // 2))
		win_eff = max(2, min(win_length, n_fft_eff))

		window = torch.hann_window(win_eff, device=signal.device, dtype=signal.dtype)

		spec = torch.stft(
			input=signal,
			n_fft=n_fft_eff,
			hop_length=hop_eff,
			win_length=win_eff,
			window=window,
			center=True,
			return_complex=True,
		)
		mag = torch.abs(spec)
		return mag

	def _spectral_convergence_loss(self, pred_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
		diff = pred_mag - target_mag
		# Frobenius norm per sample and average over batch.
		numerator = torch.linalg.norm(diff.reshape(diff.size(0), -1), ord=2, dim=1)
		denominator = torch.linalg.norm(target_mag.reshape(target_mag.size(0), -1), ord=2, dim=1)
		return (numerator / (denominator + self.eps)).mean()

	def _spectral_log_magnitude_loss(self, pred_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
		return F.l1_loss(
			torch.log(pred_mag + self.eps),
			torch.log(target_mag + self.eps),
		)

	def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		_validate_edc_shape(pred, "pred")
		_validate_edc_shape(target, "target")

		pred_1d = pred.squeeze(1)
		target_1d = target.squeeze(1)

		total_loss = pred.new_tensor(0.0)
		for n_fft, hop, win_len in self.resolutions:
			pred_mag = self._stft_magnitude(pred_1d, n_fft=n_fft, hop_length=hop, win_length=win_len)
			target_mag = self._stft_magnitude(target_1d, n_fft=n_fft, hop_length=hop, win_length=win_len)

			sc_loss = self._spectral_convergence_loss(pred_mag, target_mag)
			sm_loss = self._spectral_log_magnitude_loss(pred_mag, target_mag)
			total_loss = total_loss + sc_loss + sm_loss

		return total_loss


class DecorLoss(nn.Module):
	def __init__(
		self,
		alpha: float = 0.0,
		beta: float = 1.0,
		fft_sizes: Sequence[int] = (64, 512, 2048, 8192),
		hop_sizes: Sequence[int] = (32, 256, 1024, 4096),
		win_lengths: Sequence[int] = (64, 512, 2048, 8192),
		eps: float = 1e-7,
	):
		super().__init__()
		self.alpha = float(alpha)
		self.beta = float(beta)
		self.time_l1 = nn.L1Loss()
		self.mrstft = MultiResolutionSTFTLoss(
			fft_sizes=fft_sizes,
			hop_sizes=hop_sizes,
			win_lengths=win_lengths,
			eps=eps,
		)

	def forward(self, pred: torch.Tensor, target: torch.Tensor):
		_validate_edc_shape(pred, "pred")
		_validate_edc_shape(target, "target")

		l1_loss = self.time_l1(pred, target)
		mrstft_loss = self.mrstft(pred, target)
		total_loss = self.alpha * l1_loss + self.beta * mrstft_loss

		return {
			"loss": total_loss,
			"l1_loss": l1_loss,
			"mrstft_loss": mrstft_loss,
		}

