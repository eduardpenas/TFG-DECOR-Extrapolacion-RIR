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


def load_mono_channel0(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=True)
    data = data.T.astype(np.float32)
    return data[0], int(sr)


def resample_signal(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return signal
    tensor = torch.from_numpy(signal).unsqueeze(0)
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    return resampler(tensor).squeeze(0).numpy().astype(np.float32)


def pad_or_truncate(signal: np.ndarray, target_len: int) -> np.ndarray:
    if signal.shape[0] > target_len:
        return signal[:target_len]
    if signal.shape[0] < target_len:
        return np.pad(signal, (0, target_len - signal.shape[0]))
    return signal


def main() -> None:
    parser = argparse.ArgumentParser(description="Figura compacta del corrimiento de pico a muestra 0.")
    parser.add_argument("--root", type=str, default="data/BIRD")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--target-sr", type=int, default=48000)
    parser.add_argument("--zoom-ms", type=float, default=50.0, help="Ventana de zoom al inicio.")
    parser.add_argument("--out", type=str, default="memoria/Figures/bird_peak_shift_zoom.png")
    args = parser.parse_args()

    files = discover_audio_files(Path(args.root))
    if not files:
        raise RuntimeError("No se encontraron audios BIRD.")

    file_path = files[args.sample_idx % len(files)]

    x_mono, sr_orig = load_mono_channel0(file_path)
    x_before = resample_signal(x_mono, sr_orig, args.target_sr)
    peak_idx = int(np.argmax(np.abs(x_before)))
    x_after = x_before[peak_idx:]

    samples_1s = int(args.target_sr)
    x_before = pad_or_truncate(x_before, samples_1s)
    x_after = pad_or_truncate(x_after, samples_1s)

    t = np.arange(samples_1s) / args.target_sr
    zoom_samples = min(int(args.target_sr * (args.zoom_ms / 1000.0)), samples_1s)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.6))
    fig.suptitle("Alineacion temporal del pico directo (BIRD)", fontsize=14, fontweight="bold")

    # Panel 1: tiempo 0-1 s
    ax = axs[0]
    ax.plot(t, x_before, label="Antes (48 kHz, sin alinear)", linewidth=1.0)
    ax.plot(t, x_after, label="Despues (alineado)", linewidth=1.0)
    ax.axvline(peak_idx / args.target_sr, linestyle="--", linewidth=1.0, alpha=0.9, label=f"Pico antes: {peak_idx} muestras")
    ax.axvline(0.0, linestyle="--", linewidth=1.0, alpha=0.9, label="Pico despues: muestra 0")
    ax.set_title("Dominio temporal normalizado (0-1 s)")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.set_xlim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    # Panel 2: zoom 0-50 ms por muestra
    ax = axs[1]
    sample_axis = np.arange(zoom_samples)
    ax.plot(sample_axis, x_before[:zoom_samples], label="Antes", linewidth=1.1)
    ax.plot(sample_axis, x_after[:zoom_samples], label="Despues", linewidth=1.1)
    if peak_idx < zoom_samples:
        ax.axvline(peak_idx, linestyle="--", linewidth=1.0, alpha=0.9, label=f"Pico antes = {peak_idx}")
    ax.axvline(0, linestyle="--", linewidth=1.0, alpha=0.9, label="Pico despues = 0")
    ax.set_title(f"Zoom inicial (0-{args.zoom_ms:.0f} ms)")
    ax.set_xlabel("Muestra (48 kHz)")
    ax.set_ylabel("Amplitud")
    ax.set_xlim(0, zoom_samples)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    details = (
        f"Archivo: {file_path.name} | SR origen: {sr_orig} Hz | SR objetivo: {args.target_sr} Hz | "
        f"Delay eliminado: {peak_idx / args.target_sr * 1000.0:.2f} ms"
    )
    fig.text(0.5, 0.01, details, ha="center", fontsize=9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.savefig(out, dpi=220)
    print(f"Figura guardada en: {out}")
    print(f"Muestra usada: {file_path}")


if __name__ == "__main__":
    main()
