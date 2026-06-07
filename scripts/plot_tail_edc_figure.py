import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio


def discover_audio_files(root: Path) -> list[Path]:
    bird_sub = root / "Bird"
    search_root = bird_sub if bird_sub.is_dir() else root
    exts = {".wav", ".flac", ".flaac"}

    files: list[Path] = []
    for p in search_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def load_multichannel(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=True)
    return data.T.astype(np.float32), int(sr)


def resample_signal(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return signal
    tensor = torch.from_numpy(signal).unsqueeze(0)
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    return resampler(tensor).squeeze(0).numpy().astype(np.float32)


def onset_align(signal: np.ndarray, threshold_ratio: float = 0.05) -> tuple[np.ndarray, int]:
    max_abs = float(np.max(np.abs(signal)))
    if max_abs <= 0.0:
        return signal, 0
    threshold = threshold_ratio * max_abs
    idx = np.flatnonzero(np.abs(signal) >= threshold)
    if idx.size == 0:
        return signal, 0
    onset_idx = int(idx[0])
    return signal[onset_idx:], onset_idx


def pad_or_truncate(signal: np.ndarray, target_len: int) -> np.ndarray:
    if signal.shape[0] > target_len:
        return signal[:target_len]
    if signal.shape[0] < target_len:
        return np.pad(signal, (0, target_len - signal.shape[0]))
    return signal


def schroeder_edc_db(signal: np.ndarray, eps: float = 1e-12, db_floor: float = -80.0) -> np.ndarray:
    energy = signal.astype(np.float64) ** 2
    rev_cumsum = np.cumsum(energy[::-1])[::-1]
    rev_cumsum = rev_cumsum / (rev_cumsum[0] + eps)
    edc_db = 10.0 * np.log10(rev_cumsum + eps)
    return np.clip(edc_db, db_floor, 0.0).astype(np.float32)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    w = int(window)
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(x, kernel, mode="same")


def linearize_edc_db(
    edc_db: np.ndarray,
    t: np.ndarray,
    fit_low_db: float = -35.0,
    fit_high_db: float = -5.0,
    db_floor: float = -80.0,
) -> np.ndarray:
    mask = (edc_db <= fit_high_db) & (edc_db >= fit_low_db)
    if np.count_nonzero(mask) < 8:
        out = np.minimum.accumulate(edc_db)
        return np.clip(out, db_floor, 0.0).astype(np.float32)

    slope, intercept = np.polyfit(t[mask], edc_db[mask], deg=1)
    line = slope * t + intercept
    line = line - line[0]
    line = np.minimum.accumulate(line)
    line = np.clip(line, db_floor, 0.0)
    return line.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Figura cola reverberante + EDC en dB.")
    parser.add_argument("--root", type=str, default="data/BIRD", help="Raiz BIRD.")
    parser.add_argument("--sample-idx", type=int, default=0, help="Indice de muestra.")
    parser.add_argument("--target-sr", type=int, default=48000, help="Frecuencia objetivo.")
    parser.add_argument("--head-ms", type=float, default=50.0, help="Duracion de head en ms.")
    parser.add_argument("--onset-threshold", type=float, default=0.05, help="Umbral onset relativo.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para canal aleatorio.")
    parser.add_argument(
        "--edc-style",
        type=str,
        default="linearized",
        choices=["raw", "smooth", "linearized"],
        help="Estilo de curva EDC a visualizar.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=801,
        help="Ventana (muestras) para suavizado moving average.",
    )
    parser.add_argument("--out", type=str, default="memoria/Figures/tail_edc_overlay.png", help="PNG de salida.")
    args = parser.parse_args()

    if not 0.0 < args.onset_threshold <= 1.0:
        raise ValueError("onset-threshold debe estar en (0, 1].")

    files = discover_audio_files(Path(args.root))
    if not files:
        raise RuntimeError("No se encontraron audios en la ruta indicada.")

    file_path = files[args.sample_idx % len(files)]
    x_mc, sr_orig = load_multichannel(file_path)

    rng = np.random.default_rng(args.seed)
    ch_idx = int(rng.integers(0, x_mc.shape[0]))
    x = x_mc[ch_idx]

    x = resample_signal(x, sr_orig, args.target_sr)
    x, onset_idx = onset_align(x, threshold_ratio=args.onset_threshold)
    x = pad_or_truncate(x, int(args.target_sr * 1.0))

    head_samples = int(args.target_sr * (args.head_ms / 1000.0))
    tail = x[head_samples:]
    t_tail = np.arange(tail.shape[0]) / args.target_sr

    edc_db_raw = schroeder_edc_db(tail)
    if args.edc_style == "raw":
        edc_db_plot = edc_db_raw
    elif args.edc_style == "smooth":
        edc_db_plot = moving_average(edc_db_raw, args.smooth_window)
        edc_db_plot = np.minimum.accumulate(edc_db_plot)
        edc_db_plot = np.clip(edc_db_plot, -80.0, 0.0).astype(np.float32)
    else:
        edc_db_plot = linearize_edc_db(edc_db_raw, t_tail)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle("Cola reverberante y curva EDC en dB", fontsize=14, fontweight="bold")

    ax1.plot(t_tail, tail, color="#1f77b4", linewidth=0.8)
    ax1.set_ylabel("Amplitud")
    ax1.set_title("1) Cola reverberante pura (oscilante)")
    ax1.grid(alpha=0.25)

    ax2.plot(t_tail, edc_db_raw, color="#ffbb78", linewidth=0.9, alpha=0.55, label="EDC original")
    ax2.plot(t_tail, edc_db_plot, color="#ff7f0e", linewidth=1.8, label=f"EDC {args.edc_style}")
    ax2.set_xlabel("Tiempo (s)")
    ax2.set_ylabel("EDC (dB)")
    ax2.set_title("2) EDC en dB (decaimiento monótono)")
    ax2.set_ylim(-80, 1)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)

    details = (
        f"Archivo: {file_path.name} | Canal aleatorio: {ch_idx} | "
        f"SR orig: {sr_orig} Hz -> {args.target_sr} Hz | "
        f"Onset: {onset_idx} muestras | head: {args.head_ms:.1f} ms"
    )
    fig.text(0.5, 0.01, details, ha="center", fontsize=9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    plt.savefig(out, dpi=220)

    print(f"Figura guardada en: {out}")
    print(f"Muestra usada: {file_path}")


if __name__ == "__main__":
    main()
