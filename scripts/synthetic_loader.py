import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
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


class SyntheticRIRDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        fixed_head_samples: int = 2400,
        fixed_target_samples: int = 45600,
        normalize_range: str = "0_1",
        max_samples: Optional[int] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.fixed_head_samples = int(fixed_head_samples)
        self.fixed_target_samples = int(fixed_target_samples)

        if normalize_range not in {"0_1", "-1_1"}:
            raise ValueError("normalize_range debe ser '0_1' o '-1_1'.")
        self.normalize_range = normalize_range

        metadata_path = self.root_dir / "metadata.csv"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"No existe metadata.csv en: {self.root_dir}")

        with metadata_path.open("r", newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))

        if max_samples is not None and max_samples > 0:
            rows = rows[: int(max_samples)]

        if not rows:
            raise RuntimeError("No hay muestras en metadata.csv")

        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _pad_or_truncate(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        current_len = tensor.numel()
        if current_len == target_len:
            return tensor
        if current_len > target_len:
            return tensor[:target_len]
        return torch.nn.functional.pad(tensor, (0, target_len - current_len))

    def __getitem__(self, idx: int):
        row = self.rows[idx]

        head_path = Path(row["head_path"])
        tail_path = Path(row["tail_path"])
        edc_tail_path = Path(row["edc_tail_path"])

        if not head_path.is_file() or not tail_path.is_file() or not edc_tail_path.is_file():
            raise FileNotFoundError(
                "Algún archivo .npy no existe para sample_id="
                f"{row.get('sample_id', idx)}"
            )

        head = torch.from_numpy(np.load(head_path).astype(np.float32))
        tail = torch.from_numpy(np.load(tail_path).astype(np.float32))
        edc_tail = torch.from_numpy(np.load(edc_tail_path).astype(np.float32))

        head = self._pad_or_truncate(head, self.fixed_head_samples)
        tail = self._pad_or_truncate(tail, self.fixed_target_samples)
        edc_tail = self._pad_or_truncate(edc_tail, self.fixed_target_samples)

        if self.normalize_range == "-1_1":
            edc_tail = (2.0 * edc_tail) - 1.0

        return {
            "input": head.unsqueeze(0),
            "target": tail.unsqueeze(0),
            "target_edc": edc_tail.unsqueeze(0),
        }
