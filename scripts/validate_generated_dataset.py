import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida un dataset sintético de RIR generado por generate_synthetic_rir_dataset.py"
    )
    parser.add_argument("--data-path", type=Path, default=Path("data/raw"), help="Ruta al dataset generado")
    parser.add_argument("--expected-rooms", type=int, default=None, help="Número esperado de muestras")
    parser.add_argument("--fs", type=int, default=48000, help="Frecuencia de muestreo")
    parser.add_argument("--head-ms", type=float, default=50.0, help="Duración esperada del head en ms")
    parser.add_argument(
        "--allow-nan-rt60",
        action="store_true",
        help="Permite RT60 NaN sin marcar error (útil en casos borde)",
    )
    return parser.parse_args()


def validate_dataset(
    data_path: Path,
    expected_rooms: int | None,
    fs: int,
    head_ms: float,
    allow_nan_rt60: bool,
) -> int:
    metadata_path = data_path / "metadata.csv"
    if not metadata_path.exists():
        print(f"ERROR: no existe {metadata_path}")
        return 1

    with metadata_path.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        print("ERROR: metadata.csv está vacío")
        return 1

    expected_head_samples = int(fs * (head_ms / 1000.0))

    errors = []
    warnings = []

    if expected_rooms is not None and len(rows) != expected_rooms:
        errors.append(
            f"Número de filas en metadata ({len(rows)}) distinto de expected-rooms ({expected_rooms})"
        )

    for idx, row in enumerate(tqdm(rows, desc="Validando muestras", unit="muestra")):
        sid = row.get("sample_id", str(idx))

        try:
            length = float(row["room_length_m"])
            width = float(row["room_width_m"])
            height = float(row["room_height_m"])
            distance = float(row["source_receiver_distance_m"])
            mean_abs = float(row["mean_absorption"])
            rt60 = float(row["rt60_estimated_s"])

            if not (3.0 <= length <= 6.0):
                errors.append(f"sample_id={sid}: room_length_m fuera de rango [3,6]")
            if not (3.0 <= width <= 6.0):
                errors.append(f"sample_id={sid}: room_width_m fuera de rango [3,6]")
            if not (2.5 <= height <= 4.0):
                errors.append(f"sample_id={sid}: room_height_m fuera de rango [2.5,4]")
            if distance < 1.0:
                errors.append(f"sample_id={sid}: distancia fuente-receptor < 1m")
            if not (0.1 <= mean_abs <= 0.6):
                errors.append(f"sample_id={sid}: mean_absorption fuera de rango [0.1,0.6]")

            if np.isnan(rt60):
                if not allow_nan_rt60:
                    errors.append(f"sample_id={sid}: RT60 es NaN")
                else:
                    warnings.append(f"sample_id={sid}: RT60 NaN")
            elif rt60 <= 0:
                errors.append(f"sample_id={sid}: RT60 <= 0")

            rir_path = Path(row["rir_path"])
            head_path = Path(row["head_path"])
            tail_path = Path(row["tail_path"])
            edc_tail_path = Path(row["edc_tail_path"])

            for p in [rir_path, head_path, tail_path, edc_tail_path]:
                if not p.exists():
                    errors.append(f"sample_id={sid}: no existe archivo {p}")

            if all(p.exists() for p in [rir_path, head_path, tail_path, edc_tail_path]):
                rir = np.load(rir_path)
                head = np.load(head_path)
                tail = np.load(tail_path)
                edc_tail = np.load(edc_tail_path)

                if head.shape[0] != expected_head_samples:
                    errors.append(
                        f"sample_id={sid}: head tiene {head.shape[0]} muestras, esperado {expected_head_samples}"
                    )
                if rir.shape[0] != head.shape[0] + tail.shape[0]:
                    errors.append(f"sample_id={sid}: rir != head + tail en número de muestras")
                if edc_tail.shape[0] != tail.shape[0]:
                    errors.append(f"sample_id={sid}: edc_tail y tail tienen longitud distinta")

                max_abs = float(np.max(np.abs(rir))) if rir.size else 0.0
                if max_abs > 1.00001:
                    errors.append(f"sample_id={sid}: RIR no normalizada, max abs = {max_abs:.6f}")

                if edc_tail.size > 1:
                    diffs = np.diff(edc_tail)
                    if np.any(diffs > 1e-7):
                        errors.append(f"sample_id={sid}: EDC tail no es monótona decreciente")
                    if not np.isfinite(edc_tail).all():
                        errors.append(f"sample_id={sid}: EDC tail contiene NaN o inf")

        except Exception as exc:
            errors.append(f"sample_id={sid}: excepción durante validación -> {type(exc).__name__}: {exc}")

    print("\n=== Resumen de validación ===")
    print(f"Total muestras: {len(rows)}")
    print(f"Errores: {len(errors)}")
    print(f"Warnings: {len(warnings)}")

    if warnings:
        print("\nPrimeros 10 warnings:")
        for msg in warnings[:10]:
            print(f"- {msg}")

    if errors:
        print("\nPrimeros 20 errores:")
        for msg in errors[:20]:
            print(f"- {msg}")
        print("\nEstado: FAIL")
        return 1

    print("\nEstado: PASS")
    return 0


def main() -> None:
    args = parse_args()
    exit_code = validate_dataset(
        data_path=args.data_path,
        expected_rooms=args.expected_rooms,
        fs=args.fs,
        head_ms=args.head_ms,
        allow_nan_rt60=args.allow_nan_rt60,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
