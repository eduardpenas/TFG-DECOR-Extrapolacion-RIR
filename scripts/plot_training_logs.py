#!/usr/bin/env python3
"""Procesa un log de entrenamiento y genera figuras de métricas por época."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RE_EPOCA_GNORM = re.compile(r"^Época\s+(\d+)/(\d+):.*?gnorm=([0-9]*\.?[0-9]+)")
RE_TRAIN = re.compile(
    r"^\[Epoch\s+(\d+)\]\s+Train\s+Loss:\s+([0-9]*\.?[0-9]+)\s+\|\s+Train\s+EDC-L1:\s+([0-9]*\.?[0-9]+)"
)
RE_VAL = re.compile(
    r"^\[Epoch\s+(\d+)\]\s+Val\s+Loss:\s+([0-9]*\.?[0-9]+)\s+\|\s+Val\s+EDC-L1:\s+([0-9]*\.?[0-9]+)"
)
ROLLING_WINDOW = 15


def parse_log_file(log_path: Path) -> pd.DataFrame:
    """Lee el log línea por línea y extrae métricas con expresiones regulares."""
    rows: dict[int, dict[str, float]] = {}

    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            match_gnorm = RE_EPOCA_GNORM.match(line)
            if match_gnorm:
                epoch = int(match_gnorm.group(1))
                gnorm = float(match_gnorm.group(3))
                rows.setdefault(epoch, {})["gnorm"] = gnorm
                continue

            match_train = RE_TRAIN.match(line)
            if match_train:
                epoch = int(match_train.group(1))
                train_loss = float(match_train.group(2))
                train_edc_l1 = float(match_train.group(3))
                row = rows.setdefault(epoch, {})
                row["train_loss"] = train_loss
                row["train_edc_l1"] = train_edc_l1
                continue

            match_val = RE_VAL.match(line)
            if match_val:
                epoch = int(match_val.group(1))
                val_loss = float(match_val.group(2))
                val_edc_l1 = float(match_val.group(3))
                row = rows.setdefault(epoch, {})
                row["val_loss"] = val_loss
                row["val_edc_l1"] = val_edc_l1

    if not rows:
        raise ValueError(
            "No se han encontrado métricas en el archivo. Verifica el formato del log."
        )

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "epoch"
    df.sort_index(inplace=True)
    return df


def establish_matplotlib_style() -> None:
    """Establece un estilo visual académico compatible con múltiples versiones."""
    for style_name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try:
            plt.style.use(style_name)
            plt.rcParams.update(
                {
                    "font.size": 11,
                    "axes.titlesize": 14,
                    "axes.labelsize": 12,
                    "legend.fontsize": 10,
                    "xtick.labelsize": 10,
                    "ytick.labelsize": 10,
                }
            )
            return
        except OSError:
            pass


def add_smoothed_columns(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Genera columnas suavizadas con media móvil para loss y EDC-L1."""
    smoothed_df = df.copy()
    smoothed_df["train_loss_smooth"] = smoothed_df["train_loss"].rolling(
        window=window, min_periods=1
    ).mean()
    smoothed_df["val_loss_smooth"] = smoothed_df["val_loss"].rolling(
        window=window, min_periods=1
    ).mean()
    smoothed_df["train_edc_l1_smooth"] = smoothed_df["train_edc_l1"].rolling(
        window=window, min_periods=1
    ).mean()
    smoothed_df["val_edc_l1_smooth"] = smoothed_df["val_edc_l1"].rolling(
        window=window, min_periods=1
    ).mean()
    return smoothed_df


def generate_train_loss_figure(df: pd.DataFrame, output_dir: Path) -> None:
    """Genera la figura de Train Loss con curva original y tendencia suavizada."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(
        df.index,
        df["train_loss"],
        color="#89C2FF",
        alpha=0.3,
        linewidth=1.8,
        label="Train Loss (original)",
    )
    ax.plot(
        df.index,
        df["train_loss_smooth"],
        color="#0D47A1",
        linewidth=2.8,
        label=f"Train Loss (media móvil {ROLLING_WINDOW})",
    )

    ax.set_title("Pérdida de entrenamiento por época")
    ax.set_xlabel("Época")
    ax.set_ylabel("Magnitud")
    ax.legend(loc="best")

    fig.savefig(output_dir / "train_loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_val_loss_figure(df: pd.DataFrame, output_dir: Path) -> None:
    """Genera la figura de Val Loss con curva original y tendencia suavizada."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(
        df.index,
        df["val_loss"],
        color="#FFBE7A",
        alpha=0.3,
        linewidth=1.8,
        label="Val Loss (original)",
    )
    ax.plot(
        df.index,
        df["val_loss_smooth"],
        color="#E65100",
        linewidth=2.8,
        label=f"Val Loss (media móvil {ROLLING_WINDOW})",
    )

    ax.set_title("Pérdida de validación por época")
    ax.set_xlabel("Época")
    ax.set_ylabel("Magnitud")
    ax.legend(loc="best")

    fig.savefig(output_dir / "val_loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_train_edc_figure(df: pd.DataFrame, output_dir: Path) -> None:
    """Genera la figura de Train EDC-L1 con curva original y tendencia suavizada."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(
        df.index,
        df["train_edc_l1"],
        color="#9ED99A",
        alpha=0.3,
        linewidth=1.8,
        label="Train EDC-L1 (original)",
    )
    ax.plot(
        df.index,
        df["train_edc_l1_smooth"],
        color="#1B5E20",
        linewidth=2.8,
        label=f"Train EDC-L1 (media móvil {ROLLING_WINDOW})",
    )

    ax.set_title("Error EDC-L1 de entrenamiento por época")
    ax.set_xlabel("Época")
    ax.set_ylabel("Error L1")
    ax.legend(loc="best")

    fig.savefig(output_dir / "train_edc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_val_edc_figure(df: pd.DataFrame, output_dir: Path) -> None:
    """Genera la figura de Val EDC-L1 con curva original y tendencia suavizada."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(
        df.index,
        df["val_edc_l1"],
        color="#FF9EA0",
        alpha=0.3,
        linewidth=1.8,
        label="Val EDC-L1 (original)",
    )
    ax.plot(
        df.index,
        df["val_edc_l1_smooth"],
        color="#B71C1C",
        linewidth=2.8,
        label=f"Val EDC-L1 (media móvil {ROLLING_WINDOW})",
    )

    ax.set_title("Error EDC-L1 de validación por época")
    ax.set_xlabel("Época")
    ax.set_ylabel("Error L1")
    ax.legend(loc="best")

    fig.savefig(output_dir / "val_edc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_gnorm_figure(df: pd.DataFrame, output_dir: Path) -> None:
    """Genera la figura de la norma del gradiente en escala logarítmica."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(df.index, df["gnorm"], color="#6A1B9A", linewidth=2.2, label="gnorm")
    ax.set_yscale("log")

    ax.set_title("Norma del gradiente en escala logarítmica")
    ax.set_xlabel("Época")
    ax.set_ylabel("Magnitud")
    ax.legend(loc="best")

    fig.savefig(output_dir / "gnorm_log.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    """Crea el parser de argumentos para ejecutar el script desde terminal."""
    parser = argparse.ArgumentParser(
        description="Procesar log de entrenamiento y generar figuras por época."
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("logs_entrenamiento_completo.txt"),
        help="Ruta al archivo de log de entrenamiento.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Figures"),
        help="Directorio donde se guardarán las figuras.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    log_path = args.log_path
    output_dir = args.output_dir

    if not log_path.exists():
        raise FileNotFoundError(f"No existe el archivo de log: {log_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    establish_matplotlib_style()
    df = parse_log_file(log_path)
    df = add_smoothed_columns(df)

    required_columns = [
        "train_loss",
        "val_loss",
        "train_edc_l1",
        "val_edc_l1",
        "gnorm",
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            "Faltan métricas necesarias para generar figuras: " + ", ".join(missing)
        )

    generate_train_loss_figure(df, output_dir)
    generate_val_loss_figure(df, output_dir)
    generate_train_edc_figure(df, output_dir)
    generate_val_edc_figure(df, output_dir)
    generate_gnorm_figure(df, output_dir)

    print(f"Figuras generadas en: {output_dir.resolve()}")
    print(f"Épocas procesadas: {len(df)}")


if __name__ == "__main__":
    main()
