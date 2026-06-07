import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import soundfile as sf
import torch
import torchaudio


ORIG_REF_SR = 16000
TARGET_REF_SR = 48000
SECONDS_TO_SHOW = 1.0


def discover_audio_files(root: Path) -> list[Path]:
    bird_sub = root / "Bird"
    search_root = bird_sub if bird_sub.is_dir() else root
    exts = {".wav", ".flac", ".flaac"}

    files: list[Path] = []
    for p in search_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def load_multichannel_audio(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=True)
    # soundfile devuelve (samples, channels) -> lo pasamos a (channels, samples)
    return data.T.astype(np.float32), int(sr)


def to_mono_channel0(multichannel: np.ndarray) -> np.ndarray:
    # En BIRD/ambisonics, el canal 0 corresponde al componente omni (W)
    return multichannel[0]


def select_random_channel(multichannel: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    n_ch = multichannel.shape[0]
    ch_idx = int(rng.integers(0, n_ch))
    return multichannel[ch_idx], ch_idx


def resample_signal(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return signal
    tensor = torch.from_numpy(signal).unsqueeze(0)
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    out = resampler(tensor).squeeze(0)
    return out.numpy().astype(np.float32)


def align_to_onset(signal: np.ndarray, threshold_ratio: float = 0.05) -> tuple[np.ndarray, int]:
    max_abs = float(np.max(np.abs(signal)))
    if max_abs <= 0.0:
        return signal, 0

    threshold = threshold_ratio * max_abs
    onset_candidates = np.flatnonzero(np.abs(signal) >= threshold)
    if onset_candidates.size == 0:
        return signal, 0

    onset_idx = int(onset_candidates[0])
    return signal[onset_idx:], onset_idx


def pad_or_truncate(signal: np.ndarray, target_len: int) -> np.ndarray:
    if signal.shape[0] > target_len:
        return signal[:target_len]
    if signal.shape[0] < target_len:
        return np.pad(signal, (0, target_len - signal.shape[0]))
    return signal


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figura de preprocesado BIRD: antes/despues (canales y muestras)."
    )
    parser.add_argument("--root", type=str, default="data/BIRD", help="Raiz del dataset BIRD.")
    parser.add_argument("--sample-idx", type=int, default=0, help="Indice de muestra en orden alfabetico.")
    parser.add_argument("--target-sr", type=int, default=48000, help="Frecuencia objetivo para remuestreo.")
    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=0.05,
        help="Umbral relativo para onset detection (0.05=5%, 0.1=10%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla para reproducibilidad de la seleccion aleatoria de canal.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="scripts/fig_bird_preprocess_before_after.png",
        help="Ruta de salida de la figura PNG.",
    )
    args = parser.parse_args()
    if not 0.0 < args.onset_threshold <= 1.0:
        raise ValueError("--onset-threshold debe estar en el rango (0, 1].")

    rng = np.random.default_rng(args.seed)

    root = Path(args.root)
    files = discover_audio_files(root)
    if not files:
        raise RuntimeError(f"No se encontraron audios en {root} ni en {root / 'Bird'}")

    file_path = files[args.sample_idx % len(files)]
    x_mc, sr_orig = load_multichannel_audio(file_path)
    n_ch, n_samples = x_mc.shape

    x_mono, mono_ch_idx = select_random_channel(x_mc, rng)
    # Curva "antes" normalizada a 1 segundo con rejilla de 16 kHz para visualizacion.
    x_mono_before_ref = resample_signal(x_mono, sr_orig, ORIG_REF_SR)
    x_mono_before_ref = pad_or_truncate(x_mono_before_ref, int(ORIG_REF_SR * SECONDS_TO_SHOW))

    x_mc_before_ref = np.zeros((n_ch, int(ORIG_REF_SR * SECONDS_TO_SHOW)), dtype=np.float32)
    for ch in range(n_ch):
        ch_ref = resample_signal(x_mc[ch], sr_orig, ORIG_REF_SR)
        x_mc_before_ref[ch] = pad_or_truncate(ch_ref, int(ORIG_REF_SR * SECONDS_TO_SHOW))

    x_mono_rs = resample_signal(x_mono, sr_orig, args.target_sr)
    x_mono_before_48k = pad_or_truncate(x_mono_rs, int(args.target_sr * SECONDS_TO_SHOW))
    x_after, onset_idx_rs = align_to_onset(x_mono_rs, threshold_ratio=args.onset_threshold)
    x_after = pad_or_truncate(x_after, int(args.target_sr * SECONDS_TO_SHOW))

    samples_before = int(ORIG_REF_SR * SECONDS_TO_SHOW)
    samples_after = int(args.target_sr * SECONDS_TO_SHOW)
    t_before_s = np.arange(samples_before) / ORIG_REF_SR
    t_after_s = np.arange(samples_after) / args.target_sr

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "BIRD RIR: preprocesado antes/despues (canal aleatorio + remuestreo + onset)",
        fontsize=13,
        fontweight="bold",
    )

    # (1) Todos los canales en dominio temporal (ms)
    ax = axs[0, 0]
    for ch in range(n_ch):
        ax.plot(t_before_s, x_mc_before_ref[ch], linewidth=0.7, alpha=0.6)
    ax.set_title("Antes: multicanal (tiempo normalizado)")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.set_xlim(0.0, 1.0)
    ax.grid(alpha=0.2)

    # (2) Canal mono antes vs despues (tiempo)
    ax = axs[0, 1]
    ax.plot(t_after_s, x_mono_before_48k, label=f"Antes (48 kHz, sin alinear)", linewidth=1.0, alpha=0.9)
    ax.plot(t_after_s, x_after, label=f"Despues (alineado, {args.target_sr} Hz)", linewidth=1.0)
    delay_ms = onset_idx_rs / args.target_sr * 1000.0
    ax.axvline(delay_ms / 1000.0, linestyle="--", linewidth=1.0, alpha=0.8, label=f"Onset antes: {delay_ms:.2f} ms")
    ax.axvline(0.0, linestyle="--", linewidth=1.0, alpha=0.8, label="Onset despues: 0 ms")
    ax.set_title("Canal mono: antes vs despues (tiempo normalizado)")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # Zoom del inicio para ver claramente el corrimiento del pico.
    axins = inset_axes(ax, width="45%", height="45%", loc="upper right")
    zoom_end_s = min(0.08, 1.0)
    mask_zoom = t_after_s <= zoom_end_s
    axins.plot(t_after_s[mask_zoom], x_mono_before_48k[mask_zoom], linewidth=0.9, alpha=0.9)
    axins.plot(t_after_s[mask_zoom], x_after[mask_zoom], linewidth=0.9)
    axins.axvline(delay_ms / 1000.0, linestyle="--", linewidth=0.8, alpha=0.8)
    axins.axvline(0.0, linestyle="--", linewidth=0.8, alpha=0.8)
    axins.set_xlim(0.0, zoom_end_s)
    axins.set_title("Zoom 0-80 ms", fontsize=8)
    axins.tick_params(axis="both", labelsize=7)

    # (3) Todos los canales por indice de muestra
    ax = axs[1, 0]
    sample_idx_before = np.arange(samples_before)
    for ch in range(n_ch):
        ax.plot(sample_idx_before, x_mc_before_ref[ch], linewidth=0.7, alpha=0.6)
    ax.set_title("Antes: multicanal (indice de muestra 0-16000)")
    ax.set_xlabel("Muestra")
    ax.set_ylabel("Amplitud")
    ax.set_xlim(0, samples_before)
    ax.grid(alpha=0.2)

    # (4) Canal mono antes/despues por indice de muestra (48 kHz)
    ax = axs[1, 1]
    sample_idx_after = np.arange(samples_after)
    ax.plot(sample_idx_after, x_mono_before_48k, label="Antes (48 kHz, sin alinear)", linewidth=1.0, alpha=0.9)
    ax.plot(sample_idx_after, x_after, label="Despues (alineado)", linewidth=1.0)
    ax.axvline(onset_idx_rs, linestyle="--", linewidth=1.0, alpha=0.8, label=f"Onset antes: muestra {onset_idx_rs}")
    ax.axvline(0, linestyle="--", linewidth=1.0, alpha=0.8, label="Onset despues: muestra 0")
    ax.set_title("Canal mono: indice de muestra 0-48000")
    ax.set_xlabel("Muestra")
    ax.set_ylabel("Amplitud")
    ax.set_xlim(0, samples_after)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    details = (
        f"Archivo: {file_path.name} | Canales: {n_ch} | "
        f"Canal mono aleatorio: {mono_ch_idx} | "
        f"SR original: {sr_orig} Hz | SR objetivo: {args.target_sr} Hz | "
        f"Onset threshold: {args.onset_threshold:.2f} | Delay eliminado: {delay_ms:.2f} ms"
    )
    fig.text(0.5, 0.01, details, ha="center", fontsize=9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out, dpi=180)
    print(f"Figura guardada en: {out}")
    print(f"Muestra usada: {file_path}")


if __name__ == "__main__":
    main()
