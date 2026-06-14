import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_metadata(data_path: Path) -> list[dict[str, str]]:
    metadata_path = data_path / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No se encontro metadata.csv en: {data_path}")

    with metadata_path.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise RuntimeError("metadata.csv esta vacio. Genera primero el dataset.")

    return rows


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2f2f2f",
            "axes.labelcolor": "#2f2f2f",
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.45,
            "font.size": 9,
            "font.family": "DejaVu Sans",
            "legend.frameon": False,
            "xtick.color": "#2f2f2f",
            "ytick.color": "#2f2f2f",
            "savefig.facecolor": "white",
        }
    )


def select_rows(rows: list[dict[str, str]], num_samples: int, seed: int) -> list[dict[str, str]]:
    if num_samples <= 0:
        raise ValueError("num-samples debe ser mayor que 0.")

    if num_samples >= len(rows):
        return rows

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(rows), size=num_samples, replace=False)
    indices = np.sort(indices)
    return [rows[int(idx)] for idx in indices]


def load_sample_arrays(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rir = np.load(Path(row["rir_path"])).astype(np.float32)
    tail = np.load(Path(row["tail_path"])).astype(np.float32)
    edc_tail = np.load(Path(row["edc_tail_path"])).astype(np.float32)
    return rir, tail, edc_tail


def add_sample_row(
    axs: np.ndarray,
    row_idx: int,
    row: dict[str, str],
    rir: np.ndarray,
    tail: np.ndarray,
    edc_tail: np.ndarray,
    fs: int,
    head_ms: float,
) -> None:
    ax_rir, ax_tail, ax_edc = axs[row_idx]

    time_rir_ms = (np.arange(rir.size) / fs) * 1000.0
    head_samples = int(fs * (head_ms / 1000.0))
    time_tail_ms = ((np.arange(tail.size) + head_samples) / fs) * 1000.0
    edc_db = 10.0 * np.log10(np.maximum(edc_tail, 1e-12))

    sample_id = int(row["sample_id"])
    room_dims = (
        float(row["room_length_m"]),
        float(row["room_width_m"]),
        float(row["room_height_m"]),
    )
    distance = float(row["source_receiver_distance_m"])
    rt60 = float(row["rt60_estimated_s"])
    profile = row.get("absorption_profile", "n/a")

    ax_rir.plot(time_rir_ms, rir, color="#1f77b4", linewidth=0.9)
    ax_rir.axvline(head_ms, color="#888888", linestyle="--", linewidth=1.0, alpha=0.8)
    ax_rir.set_title(
        f"sample {sample_id:05d} | sala {room_dims[0]:.1f}x{room_dims[1]:.1f}x{room_dims[2]:.1f} m"
    )
    ax_rir.set_ylabel("Amplitud")
    ax_rir.set_xlabel("Tiempo (ms)")

    ax_tail.plot(time_tail_ms, tail, color="#ff7f0e", linewidth=0.8)
    ax_tail.set_title(f"Tail | d(S,R)={distance:.2f} m | perfil={profile}")
    ax_tail.set_ylabel("Amplitud")
    ax_tail.set_xlabel("Tiempo (ms)")

    ax_edc.plot(time_tail_ms, edc_db, color="#2ca02c", linewidth=1.0)
    ax_edc.set_title(f"EDC tail | RT60={rt60:.2f} s")
    ax_edc.set_ylabel("EDC (dB)")
    ax_edc.set_xlabel("Tiempo (ms)")
    ax_edc.set_ylim(-80, 1)


def plot_gallery(
    rows: list[dict[str, str]],
    out_path: Path,
    fs: int,
    head_ms: float,
    seed: int,
    data_path: Path,
) -> None:
    selected_rows = select_rows(rows, num_samples=len(rows), seed=seed)
    num_rows = len(selected_rows)

    fig, axs = plt.subplots(num_rows, 3, figsize=(16, max(3.2 * num_rows, 5.5)), squeeze=False)
    fig.suptitle(
        "Galeria de RIRs sinteticas aleatorias",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    for row_idx, row in enumerate(selected_rows):
        rir, tail, edc_tail = load_sample_arrays(row)
        add_sample_row(axs, row_idx, row, rir, tail, edc_tail, fs=fs, head_ms=head_ms)

    fig.text(
        0.5,
        0.008,
        f"Dataset: {data_path} | muestras mostradas: {num_rows} | fs={fs} Hz | head={head_ms:.1f} ms | seed={seed}",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera una galeria con varias RIRs aleatorias del dataset sintetico."
    )
    parser.add_argument("--data-path", type=Path, default=Path("data/raw_train_sintetico_v1"), help="Ruta al dataset sintetico")
    parser.add_argument("--num-samples", type=int, default=6, help="Numero de salas aleatorias a visualizar")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para la seleccion aleatoria")
    parser.add_argument("--fs", type=int, default=48000, help="Frecuencia de muestreo en Hz")
    parser.add_argument("--head-ms", type=float, default=50.0, help="Duracion del head en ms")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("Figures/synthetic_rooms/synthetic_rir_gallery.png"),
        help="PNG de salida",
    )
    return parser.parse_args()


def main() -> None:
    configure_style()
    args = parse_args()
    rows = load_metadata(args.data_path)
    selected_rows = select_rows(rows, num_samples=min(args.num_samples, len(rows)), seed=args.seed)
    plot_gallery(
        rows=selected_rows,
        out_path=args.out,
        fs=args.fs,
        head_ms=args.head_ms,
        seed=args.seed,
        data_path=args.data_path,
    )
    print(f"Figura guardada en: {args.out}")
    print("Sample IDs mostrados:", ", ".join(row["sample_id"] for row in selected_rows))


if __name__ == "__main__":
    main()