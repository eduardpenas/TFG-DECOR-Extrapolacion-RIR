import argparse
import csv
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
from tqdm import tqdm


CENTER_FREQS = [125, 250, 500, 1000, 2000, 4000]
WALL_NAMES = ["east", "west", "north", "south", "ceiling", "floor"]


def sample_room_dimensions(rng: np.random.Generator) -> np.ndarray:
    length = rng.uniform(3.0, 6.0)
    width = rng.uniform(3.0, 6.0)
    height = rng.uniform(2.5, 4.0)
    return np.array([length, width, height], dtype=np.float64)


def sample_band_absorption(
    rng: np.random.Generator,
    low: float = 0.1,
    high: float = 0.6,
) -> tuple[dict[str, pra.Material], np.ndarray]:
    materials = {}
    all_coeffs = []
    for wall in WALL_NAMES:
        coeffs = rng.uniform(low, high, size=len(CENTER_FREQS)).tolist()
        all_coeffs.extend(coeffs)
        materials[wall] = pra.Material(
            energy_absorption={"coeffs": coeffs, "center_freqs": CENTER_FREQS}
        )
    return materials, np.asarray(all_coeffs, dtype=np.float64)


def random_point_in_room(
    room_dims: np.ndarray,
    rng: np.random.Generator,
    margin: float = 0.2,
) -> np.ndarray:
    low = np.full(3, margin)
    high = room_dims - margin
    if np.any(high <= low):
        raise ValueError(
            "Las dimensiones de la sala no dejan margen suficiente para ubicar puntos."
        )
    return rng.uniform(low, high)


def sample_source_receiver_positions(
    room_dims: np.ndarray,
    rng: np.random.Generator,
    min_distance: float = 1.0,
    margin: float = 0.2,
    max_tries: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(max_tries):
        source = random_point_in_room(room_dims, rng=rng, margin=margin)
        receiver = random_point_in_room(room_dims, rng=rng, margin=margin)
        if np.linalg.norm(source - receiver) >= min_distance:
            return source, receiver
    raise RuntimeError(
        "No se pudo muestrear una pareja fuente-receptor con distancia mínima requerida."
    )


def simulate_rir(
    room_dims: np.ndarray,
    materials: dict[str, pra.Material],
    source: np.ndarray,
    receiver: np.ndarray,
    fs: int = 48000,
    max_order: int = 20,
) -> np.ndarray:
    room = pra.ShoeBox(
        room_dims,
        fs=fs,
        materials=materials,
        max_order=max_order,
    )
    room.add_source(source, signal=np.array([1.0], dtype=np.float64))
    room.add_microphone_array(np.c_[receiver])
    room.compute_rir()
    rir = np.asarray(room.rir[0][0], dtype=np.float32)
    return rir


def normalize_rir(rir: np.ndarray) -> np.ndarray:
    max_abs = np.max(np.abs(rir))
    if max_abs > 0:
        return rir / max_abs
    return rir


def split_head_tail(rir: np.ndarray, fs: int = 48000, head_ms: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    head_samples = int(fs * (head_ms / 1000.0))
    head = rir[:head_samples]
    tail = rir[head_samples:]
    return head, tail


def schroeder_edc(signal: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if signal.size == 0:
        return np.zeros(1, dtype=np.float32)
    energy = np.square(signal, dtype=np.float64)
    edc = np.cumsum(energy[::-1])[::-1]
    edc /= (edc[0] + eps)
    return edc.astype(np.float32)


def estimate_rt60_from_rir(rir: np.ndarray, fs: int = 48000) -> float:
    edc = schroeder_edc(rir).astype(np.float64)
    edc_db = 10.0 * np.log10(np.maximum(edc, 1e-12))
    t = np.arange(edc_db.size, dtype=np.float64) / fs

    idx_5 = np.where(edc_db <= -5.0)[0]
    idx_35 = np.where(edc_db <= -35.0)[0]
    if idx_5.size == 0 or idx_35.size == 0:
        return float("nan")

    start = idx_5[0]
    end = idx_35[0]
    if end <= start + 4:
        return float("nan")

    x = t[start : end + 1]
    y = edc_db[start : end + 1]
    slope, _ = np.polyfit(x, y, deg=1)
    if slope >= 0:
        return float("nan")

    rt60 = -60.0 / slope
    return float(rt60)


def ensure_output_structure(output_dir: Path) -> dict[str, Path]:
    subdirs = {
        "rirs": output_dir / "rirs",
        "head": output_dir / "head",
        "tail": output_dir / "tail",
        "edc_tail": output_dir / "edc_tail",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    return subdirs


def generate_dataset(
    num_rooms: int = 6000,
    output_dir: Path = Path("data/raw"),
    fs: int = 48000,
    max_order: int = 20,
    max_retries_per_room: int = 5,
    seed: int | None = 42,
) -> tuple[Path, Path, int]:
    rng = np.random.default_rng(seed)
    subdirs = ensure_output_structure(output_dir)
    metadata_path = output_dir / "metadata.csv"
    errors_path = output_dir / "generation_errors.csv"

    fieldnames = [
        "sample_id",
        "room_length_m",
        "room_width_m",
        "room_height_m",
        "source_x_m",
        "source_y_m",
        "source_z_m",
        "receiver_x_m",
        "receiver_y_m",
        "receiver_z_m",
        "source_receiver_distance_m",
        "mean_absorption",
        "rt60_estimated_s",
        "rir_path",
        "head_path",
        "tail_path",
        "edc_tail_path",
    ]

    error_fieldnames = [
        "sample_id",
        "attempt",
        "error_type",
        "error_message",
    ]

    with (
        metadata_path.open("w", newline="", encoding="utf-8") as csv_file,
        errors_path.open("w", newline="", encoding="utf-8") as errors_file,
    ):
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        errors_writer = csv.DictWriter(errors_file, fieldnames=error_fieldnames)
        errors_writer.writeheader()

        skipped_samples = 0
        sample_idx = 0

        progress_bar = tqdm(total=num_rooms, desc="Generando salas sintéticas", unit="sala")
        while sample_idx < num_rooms:
            generated_ok = False

            for attempt in range(1, max_retries_per_room + 1):
                try:
                    room_dims = sample_room_dimensions(rng)
                    materials, all_coeffs = sample_band_absorption(rng)
                    source, receiver = sample_source_receiver_positions(room_dims, rng)

                    rir = simulate_rir(
                        room_dims=room_dims,
                        materials=materials,
                        source=source,
                        receiver=receiver,
                        fs=fs,
                        max_order=max_order,
                    )
                    rir = normalize_rir(rir)
                    head, tail = split_head_tail(rir, fs=fs, head_ms=50.0)
                    edc_tail = schroeder_edc(tail)
                    rt60 = estimate_rt60_from_rir(rir, fs=fs)

                    sample_name = f"sample_{sample_idx:05d}"
                    rir_path = subdirs["rirs"] / f"{sample_name}.npy"
                    head_path = subdirs["head"] / f"{sample_name}.npy"
                    tail_path = subdirs["tail"] / f"{sample_name}.npy"
                    edc_tail_path = subdirs["edc_tail"] / f"{sample_name}.npy"

                    np.save(rir_path, rir.astype(np.float32))
                    np.save(head_path, head.astype(np.float32))
                    np.save(tail_path, tail.astype(np.float32))
                    np.save(edc_tail_path, edc_tail.astype(np.float32))

                    writer.writerow(
                        {
                            "sample_id": sample_idx,
                            "room_length_m": room_dims[0],
                            "room_width_m": room_dims[1],
                            "room_height_m": room_dims[2],
                            "source_x_m": source[0],
                            "source_y_m": source[1],
                            "source_z_m": source[2],
                            "receiver_x_m": receiver[0],
                            "receiver_y_m": receiver[1],
                            "receiver_z_m": receiver[2],
                            "source_receiver_distance_m": float(np.linalg.norm(source - receiver)),
                            "mean_absorption": float(np.mean(all_coeffs)),
                            "rt60_estimated_s": rt60,
                            "rir_path": str(rir_path),
                            "head_path": str(head_path),
                            "tail_path": str(tail_path),
                            "edc_tail_path": str(edc_tail_path),
                        }
                    )

                    generated_ok = True
                    sample_idx += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix({"errores": skipped_samples})
                    break

                except Exception as exc:
                    errors_writer.writerow(
                        {
                            "sample_id": sample_idx,
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )

            if not generated_ok:
                skipped_samples += 1
                progress_bar.set_postfix({"errores": skipped_samples})

        progress_bar.close()

    return metadata_path, errors_path, skipped_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera dataset sintético de RIRs shoebox (Muhammad & Schuller style)."
    )
    parser.add_argument("--num-rooms", type=int, default=6000, help="Número de salas a generar")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Carpeta base de salida para los .npy y metadata.csv",
    )
    parser.add_argument("--fs", type=int, default=48000, help="Frecuencia de muestreo")
    parser.add_argument(
        "--max-order",
        type=int,
        default=20,
        help="Orden máximo de reflexiones para el método de fuentes virtuales",
    )
    parser.add_argument(
        "--max-retries-per-room",
        type=int,
        default=5,
        help="Reintentos máximos por muestra antes de registrar error y continuar",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria (usa un entero o elimina este argumento para no fijar semilla)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path, errors_path, skipped_samples = generate_dataset(
        num_rooms=args.num_rooms,
        output_dir=args.output_dir,
        fs=args.fs,
        max_order=args.max_order,
        max_retries_per_room=args.max_retries_per_room,
        seed=args.seed,
    )
    print(f"Dataset generado correctamente. Metadata en: {metadata_path}")
    print(f"Log de errores en: {errors_path}")
    print(f"Muestras con error tras reintentos: {skipped_samples}")


if __name__ == "__main__":
    main()
