import argparse
import csv
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
from tqdm import tqdm


CENTER_FREQS = [125, 250, 500, 1000, 2000, 4000]
WALL_NAMES = ["east", "west", "north", "south", "ceiling", "floor"]
ABSORPTION_PROFILES = {
    "neutral": np.array([1.00, 1.00, 1.00, 1.00, 1.00, 1.00], dtype=np.float64),
    "hf_absorbente": np.array([0.75, 0.82, 0.92, 1.05, 1.20, 1.32], dtype=np.float64),
    "lf_absorbente": np.array([1.32, 1.20, 1.05, 0.92, 0.82, 0.75], dtype=np.float64),
    "medio_absorbente": np.array([0.88, 1.08, 1.20, 1.20, 1.05, 0.88], dtype=np.float64),
}


def sample_room_dimensions(rng: np.random.Generator) -> np.ndarray:
    length = rng.uniform(3.0, 6.0)
    width = rng.uniform(3.0, 6.0)
    height = rng.uniform(2.5, 4.0)
    return np.array([length, width, height], dtype=np.float64)


def sample_band_absorption(
    rng: np.random.Generator,
    low: float = 0.1,
    high: float = 0.6,
    mode: str = "uniform",
    wall_variability: float = 0.08,
    band_variability: float = 0.05,
) -> tuple[dict[str, pra.Material], dict[str, np.ndarray], str]:
    if not (0.0 <= low < high <= 1.0):
        raise ValueError("Los límites de absorción deben cumplir 0 <= low < high <= 1.")

    if mode not in {"uniform", "profiled"}:
        raise ValueError("mode debe ser 'uniform' o 'profiled'.")

    materials = {}
    wall_coeffs: dict[str, np.ndarray] = {}
    all_coeffs = []

    if mode == "uniform":
        profile_name = "uniform"
        base_curve = None
    else:
        profile_name = str(rng.choice(list(ABSORPTION_PROFILES.keys())))
        profile_shape = ABSORPTION_PROFILES[profile_name]
        center = rng.uniform(low, high)
        spread = max((high - low) * 0.35, 1e-4)
        base_curve = np.clip(
            center + rng.normal(0.0, spread / 3.0, size=len(CENTER_FREQS)),
            low,
            high,
        )
        base_curve = np.clip(base_curve * profile_shape, low, high)

    for wall in WALL_NAMES:
        if mode == "uniform":
            coeffs_arr = rng.uniform(low, high, size=len(CENTER_FREQS))
        else:
            wall_offset = rng.normal(0.0, wall_variability * (high - low))
            band_noise = rng.normal(0.0, band_variability * (high - low), size=len(CENTER_FREQS))
            coeffs_arr = np.clip(base_curve + wall_offset + band_noise, low, high)

        coeffs = coeffs_arr.tolist()
        wall_coeffs[wall] = np.asarray(coeffs_arr, dtype=np.float64)
        all_coeffs.extend(coeffs)
        materials[wall] = pra.Material(
            energy_absorption={"coeffs": coeffs, "center_freqs": CENTER_FREQS}
        )
    return materials, wall_coeffs, profile_name


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
    absorption_low: float = 0.1,
    absorption_high: float = 0.6,
    absorption_mode: str = "uniform",
    wall_variability: float = 0.08,
    band_variability: float = 0.05,
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
        "absorption_profile",
        "mean_absorption",
        "absorption_center_freqs_hz",
        "absorption_east_coeffs",
        "absorption_west_coeffs",
        "absorption_north_coeffs",
        "absorption_south_coeffs",
        "absorption_ceiling_coeffs",
        "absorption_floor_coeffs",
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
                    materials, wall_coeffs, profile_name = sample_band_absorption(
                        rng,
                        low=absorption_low,
                        high=absorption_high,
                        mode=absorption_mode,
                        wall_variability=wall_variability,
                        band_variability=band_variability,
                    )
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
                    all_coeffs = np.concatenate([wall_coeffs[wall] for wall in WALL_NAMES], dtype=np.float64)

                    def _format_coeffs(values: np.ndarray) -> str:
                        return ",".join(f"{float(value):.6f}" for value in values)

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
                            "absorption_profile": profile_name,
                            "mean_absorption": float(np.mean(all_coeffs)),
                            "absorption_center_freqs_hz": ",".join(str(freq) for freq in CENTER_FREQS),
                            "absorption_east_coeffs": _format_coeffs(wall_coeffs["east"]),
                            "absorption_west_coeffs": _format_coeffs(wall_coeffs["west"]),
                            "absorption_north_coeffs": _format_coeffs(wall_coeffs["north"]),
                            "absorption_south_coeffs": _format_coeffs(wall_coeffs["south"]),
                            "absorption_ceiling_coeffs": _format_coeffs(wall_coeffs["ceiling"]),
                            "absorption_floor_coeffs": _format_coeffs(wall_coeffs["floor"]),
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
    parser.add_argument(
        "--absorption-low",
        type=float,
        default=0.1,
        help="Límite inferior del coeficiente de absorción por banda (0..1)",
    )
    parser.add_argument(
        "--absorption-high",
        type=float,
        default=0.6,
        help="Límite superior del coeficiente de absorción por banda (0..1)",
    )
    parser.add_argument(
        "--absorption-mode",
        type=str,
        choices=["uniform", "profiled"],
        default="uniform",
        help="Modo de muestreo de absorción: uniforme o perfilado por frecuencia",
    )
    parser.add_argument(
        "--wall-variability",
        type=float,
        default=0.08,
        help="Variabilidad entre paredes en modo profiled (escala relativa)",
    )
    parser.add_argument(
        "--band-variability",
        type=float,
        default=0.05,
        help="Ruido entre bandas en modo profiled (escala relativa)",
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
        absorption_low=args.absorption_low,
        absorption_high=args.absorption_high,
        absorption_mode=args.absorption_mode,
        wall_variability=args.wall_variability,
        band_variability=args.band_variability,
    )
    print(f"Dataset generado correctamente. Metadata en: {metadata_path}")
    print(f"Log de errores en: {errors_path}")
    print(f"Muestras con error tras reintentos: {skipped_samples}")


if __name__ == "__main__":
    main()
