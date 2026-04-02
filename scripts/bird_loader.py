import argparse
import os
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
        fixed_target_samples: int = 48000,
        normalize_range: str = "0_1",
        audio_extensions: Sequence[str] = (".wav", ".flac", ".flaac"),
    ):
        self.root_dir = Path(root_dir)
        self.folds = self._normalize_folds(folds)
        self.head_ms = float(head_ms)
        self.fixed_head_samples = int(fixed_head_samples)
        self.fixed_target_samples = int(fixed_target_samples)
        self.audio_extensions = tuple(ext.lower() for ext in audio_extensions)

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

        if self.folds:
            for fold in self.folds:
                fold_path = self.root_dir / fold
                if not fold_path.exists() or not fold_path.is_dir():
                    continue
                file_paths.extend(self._walk_audio_files(fold_path))
            return file_paths

        file_paths.extend(self._walk_audio_files(self.root_dir))

        if file_paths:
            return file_paths

        for candidate in (self.root_dir / "Bird", self.root_dir / "link_to_folds"):
            if candidate.exists() and candidate.is_dir():
                file_paths.extend(self._walk_audio_files(candidate))
                if file_paths:
                    break

        return file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def _to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        return waveform

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

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        waveform, sample_rate = self._load_audio(file_path)
        waveform = self._to_mono(waveform).to(torch.float32)

        head_samples = max(1, int(round(sample_rate * self.head_ms / 1000.0)))
        head = waveform[:head_samples]
        tail = waveform[head_samples:]

        head = self._pad_or_truncate(head, self.fixed_head_samples)
        tail = self._pad_or_truncate(tail, self.fixed_target_samples)
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
