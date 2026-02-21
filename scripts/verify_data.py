import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def verify_generated_sample(data_path: Path, sample_id: int = 0, fs: int = 48000, show: bool = False) -> None:
    metadata_path = data_path / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No se encontró metadata.csv en: {data_path}")

    with metadata_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    if not rows:
        raise RuntimeError("metadata.csv está vacío. Genera primero el dataset.")

    row = None
    for row_candidate in rows:
        if int(row_candidate["sample_id"]) == sample_id:
            row = row_candidate
            break

    if row is None:
        raise ValueError(f"No existe sample_id={sample_id} en metadata.csv")

    rir = np.load(Path(row["rir_path"]))
    head = np.load(Path(row["head_path"]))
    tail = np.load(Path(row["tail_path"]))
    edc_tail = np.load(Path(row["edc_tail_path"]))

    print("=== Muestra seleccionada ===")
    print(f"sample_id: {int(row['sample_id'])}")
    print(
        "dimensiones sala (m): "
        f"({float(row['room_length_m']):.2f}, {float(row['room_width_m']):.2f}, {float(row['room_height_m']):.2f})"
    )
    print(f"distancia fuente-receptor (m): {float(row['source_receiver_distance_m']):.2f}")
    print(f"RT60 estimado (s): {row['rt60_estimated_s']}")
    print(f"RIR shape: {rir.shape}, Head shape: {head.shape}, Tail shape: {tail.shape}, EDC shape: {edc_tail.shape}")

    time_rir_ms = (np.arange(rir.size) / fs) * 1000.0
    time_head_ms = (np.arange(head.size) / fs) * 1000.0
    time_tail_ms = ((np.arange(tail.size) + head.size) / fs) * 1000.0

    fig, axs = plt.subplots(4, 1, figsize=(12, 10))

    axs[0].plot(time_rir_ms, rir)
    axs[0].set_title("RIR completa normalizada")
    axs[0].set_xlabel("Tiempo (ms)")
    axs[0].set_ylabel("Amplitud")

    axs[1].plot(time_head_ms, head)
    axs[1].set_title("Head (0-50 ms)")
    axs[1].set_xlabel("Tiempo (ms)")
    axs[1].set_ylabel("Amplitud")

    axs[2].plot(time_tail_ms, tail, color="orange")
    axs[2].set_title("Tail (>50 ms)")
    axs[2].set_xlabel("Tiempo (ms)")
    axs[2].set_ylabel("Amplitud")

    axs[3].plot(time_tail_ms, 10.0 * np.log10(np.maximum(edc_tail, 1e-12)), color="green")
    axs[3].set_title("EDC tail (dB)")
    axs[3].set_xlabel("Tiempo (ms)")
    axs[3].set_ylabel("dB")

    plt.tight_layout()
    output_figure = Path("scripts/verificacion_data.png")
    plt.savefig(output_figure)
    print(f"Figura guardada en: {output_figure}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verifica y visualiza muestras del dataset sintético generado.")
    parser.add_argument("--data-path", type=Path, default=Path("data/raw"), help="Ruta al dataset generado")
    parser.add_argument("--sample-id", type=int, default=0, help="ID de muestra a visualizar")
    parser.add_argument("--fs", type=int, default=48000, help="Frecuencia de muestreo en Hz")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Muestra la figura en pantalla además de guardarla (requiere entorno gráfico)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    verify_generated_sample(data_path=args.data_path, sample_id=args.sample_id, fs=args.fs, show=args.show)