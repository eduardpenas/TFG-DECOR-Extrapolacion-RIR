import argparse
import os
import random
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset


def schroeder_integration(
    signal: torch.Tensor,
    eps: float = 1e-12,
    db_floor: float = -80.0,
    normalize_range: str = "0_1",
) -> torch.Tensor:
    if signal.numel() == 0:
        return torch.zeros(1, dtype=torch.float32)

    signal = signal.to(torch.float64)
    energy = signal.pow(2)
    reversed_energy = torch.flip(energy, dims=(0,))
    reversed_cumsum = torch.cumsum(reversed_energy, dim=0)
    edc_linear = torch.flip(reversed_cumsum, dims=(0,))
    edc_linear = edc_linear / (edc_linear[0] + eps)

    edc_db = 10.0 * torch.log10(edc_linear + eps)
    edc_db = torch.clamp(edc_db, min=db_floor, max=0.0)

    if normalize_range == "-1_1":
        edc_norm = 2.0 * (edc_db - db_floor) / (0.0 - db_floor) - 1.0
    else:
        edc_norm = (edc_db - db_floor) / (0.0 - db_floor)

    return edc_norm.to(torch.float32)


class BirdDataset(Dataset):
    def __init__(
        self,
        root_dir: Union[str, Path],
        folds: Optional[Sequence[Union[int, str]]] = None,
        head_ms: float = 50.0,
        fixed_head_samples: int = 2400,
        fixed_target_samples: int = 45600,
        normalize_range: str = "0_1",
        audio_extensions: Sequence[str] = (".wav", ".flac", ".flaac"),
        target_sample_rate: int = 48000,
        augment: bool = False,
        channel_select: str = "omni",
    ):
        self.root_dir = Path(root_dir)
        self.folds = self._normalize_folds(folds)
        self.head_ms = float(head_ms)
        self.fixed_head_samples = int(fixed_head_samples)
        self.fixed_target_samples = int(fixed_target_samples)
        self.audio_extensions = tuple(ext.lower() for ext in audio_extensions)
        self.target_sample_rate = int(target_sample_rate)
        self.augment = bool(augment)
        self._resamplers: dict = {}

        if channel_select not in {"omni", "random", "mean"}:
            raise ValueError("channel_select debe ser 'omni', 'random' o 'mean'.")
        self.channel_select = channel_select

        if normalize_range not in {"0_1", "-1_1"}:
            raise ValueError("normalize_range debe ser '0_1' o '-1_1'.")
        self.normalize_range = normalize_range

        self.file_paths = self._collect_audio_paths()
        if not self.file_paths:
            raise RuntimeError(
                "No se encontraron audios compatibles "
                f"({', '.join(self.audio_extensions)}) en {self.root_dir} "
                f"con folds={self.folds if self.folds else 'AUTO'}"
            )

    def _normalize_folds(
        self, folds: Optional[Sequence[Union[int, str]]]
    ) -> List[str]:
        if folds is None:
            return []

        normalized_folds: List[str] = []
        for fold in folds:
            if isinstance(fold, int):
                normalized_folds.append(f"fold{fold:03d}")
                continue

            fold_str = str(fold).strip().lower()
            if fold_str.isdigit():
                normalized_folds.append(f"fold{int(fold_str):03d}")
            elif fold_str.startswith("fold") and fold_str[4:].isdigit():
                normalized_folds.append(f"fold{int(fold_str[4:]):03d}")
            else:
                normalized_folds.append(str(fold).strip())

        return sorted(set(normalized_folds))

    def _is_audio_file(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in self.audio_extensions

    def _walk_audio_files(self, base_dir: Path) -> List[Path]:
        discovered: List[Path] = []
        for current_root, _, filenames in os.walk(base_dir, followlinks=True):
            root_path = Path(current_root)
            for filename in filenames:
                file_path = root_path / filename
                if self._is_audio_file(file_path):
                    discovered.append(file_path)
        return sorted(discovered)

    def _collect_audio_paths(self) -> List[Path]:
        file_paths: List[Path] = []

        # Resolver la raíz efectiva: si existe Bird/ como subdirectorio, buscar folds ahí.
        bird_sub = self.root_dir / "Bird"
        effective_root = bird_sub if (bird_sub.exists() and bird_sub.is_dir()) else self.root_dir

        if self.folds:
            for fold in self.folds:
                fold_path = effective_root / fold
                if not fold_path.exists() or not fold_path.is_dir():
                    continue
                file_paths.extend(self._walk_audio_files(fold_path))
            return file_paths

        file_paths.extend(self._walk_audio_files(effective_root))
        return file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def _to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """Selecciona un canal según channel_select:
          'omni'  : canal 0 = W omnidireccional (ambisonics / BIRD).
          'random': canal aleatorio (multicanal no ambiónico).
          'mean'  : promedio de todos los canales.
        """
        if waveform.dim() == 1:
            return waveform
        n_ch = waveform.shape[0]
        if n_ch == 1:
            return waveform[0]
        if self.channel_select == "omni":
            return waveform[0]
        if self.channel_select == "random":
            ch = random.randint(0, n_ch - 1)
            return waveform[ch]
        # mean
        return waveform.mean(dim=0)

    def _load_audio(self, file_path: Path) -> tuple[torch.Tensor, int]:
        try:
            waveform, sample_rate = torchaudio.load(str(file_path))
            return waveform, int(sample_rate)
        except Exception:
            try:
                import soundfile as sf

                audio_np, sample_rate = sf.read(str(file_path), always_2d=True)
                audio_np = np.asarray(audio_np, dtype=np.float32).T
                waveform = torch.from_numpy(audio_np)
                return waveform, int(sample_rate)
            except Exception as soundfile_exc:
                raise RuntimeError(
                    f"No se pudo leer audio en {file_path}. "
                    "Si es .flaac, renómbralo/convierte a .flac válido. "
                    "Para torchaudio>=2.10 puede requerirse torchcodec+ffmpeg o fallback soundfile."
                ) from soundfile_exc

    def _pad_or_truncate(self, tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        current_len = tensor.numel()
        if current_len == target_len:
            return tensor
        if current_len > target_len:
            return tensor[:target_len]
        return F.pad(tensor, (0, target_len - current_len))

    def _get_resampler(self, orig_sr: int) -> torchaudio.transforms.Resample:
        """Devuelve (y cachea) un Resample orig_sr → target_sample_rate."""
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=self.target_sample_rate
            )
        return self._resamplers[orig_sr]

    def _apply_lowpass(self, waveform: torch.Tensor) -> torch.Tensor:
        """Filtro paso-bajo con cutoff aleatorio de {8k,12k,16k,22.05k,24k} Hz (paper §3.2)."""
        cutoffs = [c for c in (8000, 12000, 16000, 22050, 24000)
                   if c < self.target_sample_rate // 2]
        if not cutoffs:
            return waveform
        cutoff = random.choice(cutoffs)
        return torchaudio.functional.lowpass_biquad(
            waveform.unsqueeze(0),
            sample_rate=self.target_sample_rate,
            cutoff_freq=float(cutoff),
        ).squeeze(0)

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        waveform, sample_rate = self._load_audio(file_path)
        waveform = self._to_mono(waveform).to(torch.float32)

        # 1. Resamplear a la frecuencia objetivo (48 kHz)
        if sample_rate != self.target_sample_rate:
            resampler = self._get_resampler(sample_rate)
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

        # 2. Eliminar delay inicial (alinear al pico de amplitud)
        peak_idx = int(waveform.abs().argmax().item())
        waveform = waveform[peak_idx:]

        # 3. Normalizar amplitud absoluta a 1.0
        max_amp = waveform.abs().max()
        if max_amp > 0:
            waveform = waveform / max_amp

        # 4. Ajustar a 1 segundo exacto (head + tail)
        total_samples = self.fixed_head_samples + self.fixed_target_samples
        waveform = self._pad_or_truncate(waveform, total_samples)

        # 5. Augmentación: filtro paso-bajo aleatorio (robustez al sr, paper §3.2)
        if self.augment:
            waveform = self._apply_lowpass(waveform)

        # 6. Separar head (0–50 ms) y tail (50 ms–1 s)
        head = waveform[:self.fixed_head_samples]
        tail = waveform[self.fixed_head_samples:]

        target_edc = schroeder_integration(tail, normalize_range=self.normalize_range)
        target_edc = self._pad_or_truncate(target_edc, self.fixed_target_samples)

        return {
            "input": head.unsqueeze(0),
            "target": tail.unsqueeze(0),
            "target_edc": target_edc.unsqueeze(0),
        }


def _parse_folds(folds_arg: str) -> Optional[List[Union[int, str]]]:
    if not folds_arg or folds_arg.strip().lower() in {"auto", "none"}:
        return None

    parsed: List[Union[int, str]] = []
    for token in folds_arg.split(","):
        token = token.strip()
        if not token:
            continue
        parsed.append(int(token) if token.isdigit() else token)

    return parsed or None


def main():
    parser = argparse.ArgumentParser(description="Sanity-check rápido del BirdDataset")
    parser.add_argument("--root-dir", type=str, default="data/BIRD", help="Raíz del dataset BIRD.")
    parser.add_argument(
        "--folds",
        type=str,
        default="auto",
        help="Lista de folds separada por comas (ej: fold001,2,3) o 'auto'.",
    )
    parser.add_argument("--normalize-range", type=str, default="0_1", choices=["0_1", "-1_1"])
    args = parser.parse_args()

    folds = _parse_folds(args.folds)
    dataset = BirdDataset(
        root_dir=args.root_dir,
        folds=folds,
        normalize_range=args.normalize_range,
    )

    print(f"Total audios encontrados: {len(dataset)}")
    sample = dataset[0]

    input_tensor = sample["input"]
    target_tail_tensor = sample["target"]
    target_edc_tensor = sample["target_edc"]

    print(
        "input:",
        f"shape={tuple(input_tensor.shape)}",
        f"dtype={input_tensor.dtype}",
        f"min={float(input_tensor.min()):.6f}",
        f"max={float(input_tensor.max()):.6f}",
    )
    print(
        "target_tail:",
        f"shape={tuple(target_tail_tensor.shape)}",
        f"dtype={target_tail_tensor.dtype}",
        f"min={float(target_tail_tensor.min()):.6f}",
        f"max={float(target_tail_tensor.max()):.6f}",
    )
    print(
        "target_edc:",
        f"shape={tuple(target_edc_tensor.shape)}",
        f"dtype={target_edc_tensor.dtype}",
        f"min={float(target_edc_tensor.min()):.6f}",
        f"max={float(target_edc_tensor.max()):.6f}",
    )


if __name__ == "__main__":
    main()
