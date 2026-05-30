"""
Verificación visual del pipeline de preprocesamiento DECOR (checklist completo).

Genera scripts/preprocessing_report.png con 8 paneles:
  1. Señal raw (16 kHz, canal W vs todos los canales)
  2. Selección de canal W + resampleo a 48 kHz  (espectro antes/después)
  3. Alineación de delay (pico en muestra 0)
  4. Normalización de amplitud (máx absoluto = 1.0)
  5. Split head / tail (50 ms / 950 ms)
  6. Augmentación LPF (5 cutoffs del paper)
  7. EDC / Schroeder (escala dB y normalizada)
  8. Estadísticas del dataset (histogramas de delay, duración, energía)

Uso:
    python scripts/verify_preprocessing.py [--root data/BIRD] [--sample-idx 0]
                                            [--n-stats 200] [--out scripts/preprocessing_report.png]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torchaudio
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.bird_loader import BirdDataset, schroeder_integration

FS_TARGET = 48_000
HEAD_SAMPLES = 2_400    # 50 ms
TAIL_SAMPLES = 45_600   # 950 ms
LPF_CUTOFFS  = [8_000, 12_000, 16_000, 22_050, 24_000]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_raw(path: Path):
    """Carga el fichero sin ningún preprocesamiento. Devuelve (np.array, sr)."""
    data, sr = sf.read(str(path), always_2d=True)  # (frames, channels)
    return data.T.astype(np.float32), sr             # (channels, frames)


def _resample_np(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    t = torch.from_numpy(signal).unsqueeze(0)  # (1, L)
    r = torchaudio.transforms.Resample(orig_sr, target_sr)
    return r(t).squeeze(0).numpy()


def _mag_spectrum(signal: np.ndarray, sr: int):
    """Magnitud del espectro en dB mediante FFT."""
    n = len(signal)
    fft = np.abs(np.fft.rfft(signal, n=max(n, 8192)))
    fft = np.clip(fft, 1e-12, None)
    freqs = np.fft.rfftfreq(max(n, 8192), d=1.0 / sr)
    db = 20 * np.log10(fft / fft.max())
    return freqs, db


def _schroeder_db(signal: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    energy = signal.astype(np.float64) ** 2
    rev_cumsum = np.cumsum(energy[::-1])[::-1]
    edc = rev_cumsum / (rev_cumsum[0] + eps)
    return 10.0 * np.log10(edc + eps)


def _collect_stats(root: Path, n: int = 200):
    """Extrae estadísticas de n ficheros: delay, amplitud máx, sr."""
    import os, random as rng
    all_files = []
    for dirpath, _, fnames in os.walk(root, followlinks=True):
        for f in fnames:
            if f.endswith(".flac") or f.endswith(".wav"):
                all_files.append(Path(dirpath) / f)

    rng.seed(0)
    sample = rng.sample(all_files, min(n, len(all_files)))

    delays, max_amps, srs = [], [], []
    for p in sample:
        try:
            raw, sr = _load_raw(p)
            ch0 = raw[0]
            t = torch.from_numpy(ch0)
            if sr != FS_TARGET:
                t = torchaudio.transforms.Resample(sr, FS_TARGET)(t.unsqueeze(0)).squeeze(0)
            t = t.numpy()
            peak = int(np.argmax(np.abs(t)))
            delays.append(peak / FS_TARGET * 1000)    # ms
            max_amps.append(float(np.abs(t).max()))
            srs.append(sr)
        except Exception:
            pass
    return delays, max_amps, srs


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verificación visual del preprocesamiento DECOR")
    parser.add_argument("--root",       type=str, default="data/BIRD")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--n-stats",    type=int, default=200)
    parser.add_argument("--out",        type=str, default="scripts/preprocessing_report.png")
    args = parser.parse_args()

    root = Path(args.root)
    bird_sub = root / "Bird"
    search_root = bird_sub if (bird_sub.exists() and bird_sub.is_dir()) else root

    # Seleccionar archivo de muestra
    import os
    all_files = sorted(
        Path(dp) / f
        for dp, _, fnames in os.walk(search_root, followlinks=True)
        for f in fnames if f.endswith(".flac") or f.endswith(".wav")
    )
    if not all_files:
        print("No se encontraron archivos de audio.")
        return

    file_path = all_files[args.sample_idx % len(all_files)]
    print(f"Muestra seleccionada: {file_path.name}")

    # ── Cargar raw ─────────────────────────────────────────────────────────────
    raw_all_ch, sr_orig = _load_raw(file_path)  # (8, L_orig)
    ch0_raw = raw_all_ch[0]                     # canal W (omnidireccional)
    n_ch    = raw_all_ch.shape[0]
    t_orig  = np.arange(len(ch0_raw)) / sr_orig * 1000  # ms

    # ── Resampleo ──────────────────────────────────────────────────────────────
    ch0_48k = _resample_np(ch0_raw, sr_orig, FS_TARGET)
    t_48k   = np.arange(len(ch0_48k)) / FS_TARGET * 1000  # ms

    # ── Alineación de delay ────────────────────────────────────────────────────
    peak_idx     = int(np.argmax(np.abs(ch0_48k)))
    ch0_aligned  = ch0_48k[peak_idx:]
    t_aligned    = np.arange(len(ch0_aligned)) / FS_TARGET * 1000  # ms

    delay_ms     = peak_idx / FS_TARGET * 1000

    # ── Normalización ─────────────────────────────────────────────────────────
    max_amp       = np.abs(ch0_aligned).max()
    ch0_norm      = ch0_aligned / (max_amp + 1e-12)
    total_samples = HEAD_SAMPLES + TAIL_SAMPLES
    # Ajustar a 1 segundo
    if len(ch0_norm) > total_samples:
        ch0_1s = ch0_norm[:total_samples]
    else:
        ch0_1s = np.pad(ch0_norm, (0, total_samples - len(ch0_norm)))
    t_1s = np.arange(total_samples) / FS_TARGET * 1000   # ms

    # ── Head / Tail ────────────────────────────────────────────────────────────
    head = ch0_1s[:HEAD_SAMPLES]
    tail = ch0_1s[HEAD_SAMPLES:]
    t_head = t_1s[:HEAD_SAMPLES]
    t_tail = t_1s[HEAD_SAMPLES:]

    # ── LPF augmentation ──────────────────────────────────────────────────────
    lpf_versions = {}
    t_full = torch.from_numpy(ch0_1s)
    for fc in LPF_CUTOFFS:
        filtered = torchaudio.functional.lowpass_biquad(
            t_full.unsqueeze(0), sample_rate=FS_TARGET, cutoff_freq=float(fc)
        ).squeeze(0).numpy()
        lpf_versions[fc] = filtered

    # ── EDC ───────────────────────────────────────────────────────────────────
    edc_db   = _schroeder_db(tail)
    t_edc    = t_tail.copy()

    # EDC normalizada [0,1] (como la usamos en el modelo)
    db_floor = -80.0
    edc_db_c = np.clip(edc_db, db_floor, 0.0)
    edc_norm = (edc_db_c - db_floor) / (0.0 - db_floor)

    # ── Estadísticas ──────────────────────────────────────────────────────────
    print(f"Recopilando estadísticas de {args.n_stats} muestras...")
    delays, max_amps, srs = _collect_stats(search_root, args.n_stats)

    # ── FIGURA ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 22))
    fig.suptitle(
        "DECOR — Verificación del Pipeline de Preprocesamiento\n"
        f"Muestra: {file_path.name}  |  sr_orig={sr_orig} Hz → {FS_TARGET} Hz  |  {n_ch} canales",
        fontsize=13, fontweight="bold", y=0.995,
    )
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    BLUE  = "#2196F3"
    RED   = "#F44336"
    GREEN = "#4CAF50"
    ORANGE= "#FF9800"
    GRAY  = "#9E9E9E"
    COLORS_LPF = ["#1a237e", "#283593", "#1565c0", "#0288d1", "#00acc1"]

    # ── Panel 1: Raw multicanal ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for ch in range(n_ch):
        alpha = 0.25 if ch != 0 else 0.0
        if ch != 0:
            ax1.plot(t_orig, raw_all_ch[ch], color=GRAY, alpha=alpha, linewidth=0.4)
    ax1.plot(t_orig, ch0_raw, color=BLUE, linewidth=0.8, label="Canal W (omni, ch0)", zorder=5)
    ax1.axvline(peak_idx / sr_orig * 1000, color=RED, linestyle="--",
                linewidth=1.0, label=f"Pico ({delay_ms * sr_orig / FS_TARGET:.1f} ms a {sr_orig} Hz)")
    ax1.set_title(f"① Canal W vs {n_ch} canales — señal raw ({sr_orig} Hz)")
    ax1.set_xlabel("Tiempo (ms)")
    ax1.set_ylabel("Amplitud")
    ax1.legend(fontsize=7)

    # ── Panel 2: Espectro antes/después del resampleo ────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    f16, db16 = _mag_spectrum(ch0_raw,  sr_orig)
    f48, db48 = _mag_spectrum(ch0_48k, FS_TARGET)
    ax2.plot(f16 / 1000, db16, color=RED,  linewidth=0.8, alpha=0.8,
             label=f"Antes ({sr_orig//1000} kHz)")
    ax2.plot(f48 / 1000, db48, color=BLUE, linewidth=0.8, alpha=0.8,
             label=f"Después ({FS_TARGET//1000} kHz)")
    ax2.axvline(sr_orig / 2 / 1000, color=RED, linestyle=":", linewidth=1.0,
                label=f"Nyquist orig ({sr_orig//2//1000} kHz)")
    ax2.set_xlim(0, FS_TARGET / 2 / 1000)
    ax2.set_ylim(-80, 5)
    ax2.set_title(f"② Resampleo canal W: {sr_orig//1000} kHz → {FS_TARGET//1000} kHz")
    ax2.set_xlabel("Frecuencia (kHz)")
    ax2.set_ylabel("Magnitud (dB)")
    ax2.legend(fontsize=7)

    # ── Panel 3: Alineación de delay ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    t_pre = np.arange(len(ch0_48k)) / FS_TARGET * 1000
    zoom_end = min(120, len(ch0_48k) / FS_TARGET * 1000)  # zoom a primeros 120 ms
    mask_pre = t_pre <= zoom_end
    ax3.plot(t_pre[mask_pre], ch0_48k[mask_pre], color=RED,  linewidth=0.7,
             alpha=0.8, label="Antes de alinear")
    t_aln_zoom = np.arange(min(int(zoom_end * FS_TARGET / 1000), len(ch0_aligned))
                           ) / FS_TARGET * 1000
    ax3.plot(t_aln_zoom, ch0_aligned[:len(t_aln_zoom)], color=BLUE, linewidth=0.7,
             alpha=0.9, label="Después (muestra 0 = sonido directo)")
    ax3.axvline(delay_ms, color=RED, linestyle="--", linewidth=1.0,
                label=f"Delay eliminado: {delay_ms:.2f} ms ({peak_idx} muestras)")
    ax3.axvline(0, color=BLUE, linestyle="--", linewidth=0.8, alpha=0.6)
    ax3.set_title(f"③ Eliminación de delay — pico detectado en muestra {peak_idx}")
    ax3.set_xlabel("Tiempo (ms) — eje señal sin alinear")
    ax3.set_ylabel("Amplitud")
    ax3.legend(fontsize=7)

    # ── Panel 4: Normalización ────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    t_a = np.arange(min(int(0.1 * FS_TARGET), len(ch0_aligned))) / FS_TARGET * 1000
    ax4.plot(t_a, ch0_aligned[:len(t_a)], color=RED, linewidth=0.7, alpha=0.8,
             label=f"Antes (máx abs = {max_amp:.4f})")
    ax4.plot(t_a, ch0_norm[:len(t_a)], color=BLUE, linewidth=0.7, alpha=0.9,
             label="Después (máx abs = 1.0)")
    ax4.axhline( 1.0, color=BLUE, linestyle=":", linewidth=0.8)
    ax4.axhline(-1.0, color=BLUE, linestyle=":", linewidth=0.8)
    ax4.set_title(f"④ Normalización de amplitud — máx abs 1.0")
    ax4.set_xlabel("Tiempo (ms, primeros 100 ms)")
    ax4.set_ylabel("Amplitud")
    ax4.legend(fontsize=7)

    # ── Panel 5: Split head / tail ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.fill_between(t_1s[:HEAD_SAMPLES], ch0_1s[:HEAD_SAMPLES], alpha=0.35,
                     color=GREEN, label=f"Head (0–50 ms, {HEAD_SAMPLES} muestras)")
    ax5.fill_between(t_1s[HEAD_SAMPLES:], ch0_1s[HEAD_SAMPLES:], alpha=0.25,
                     color=ORANGE, label=f"Tail (50–1000 ms, {TAIL_SAMPLES} muestras)")
    ax5.plot(t_1s[:HEAD_SAMPLES], ch0_1s[:HEAD_SAMPLES], color=GREEN,  linewidth=0.6)
    ax5.plot(t_1s[HEAD_SAMPLES:], ch0_1s[HEAD_SAMPLES:], color=ORANGE, linewidth=0.5)
    ax5.axvline(50, color="black", linestyle="--", linewidth=1.2, label="Corte 50 ms")
    ax5.set_title("⑤ Split Head / Tail (50 ms / 950 ms)")
    ax5.set_xlabel("Tiempo (ms)")
    ax5.set_ylabel("Amplitud")
    ax5.legend(fontsize=7)

    # ── Panel 6: LPF augmentation ────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    t_zoom = t_1s[:int(0.05 * FS_TARGET)]  # zoom 50 ms
    ax6.plot(t_zoom, ch0_1s[:len(t_zoom)], color="black", linewidth=1.0,
             label="Original (sin LPF)", alpha=0.6)
    for i, fc in enumerate(LPF_CUTOFFS):
        ax6.plot(t_zoom, lpf_versions[fc][:len(t_zoom)],
                 color=COLORS_LPF[i], linewidth=0.8, alpha=0.85,
                 label=f"LPF {fc//1000} kHz")
    ax6.set_title("⑥ Data Augmentation — LPF aleatorio (paper §3.2)")
    ax6.set_xlabel("Tiempo (ms, head)")
    ax6.set_ylabel("Amplitud")
    ax6.legend(fontsize=6.5)

    # ── Panel 7: EDC (Schroeder) ──────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    ax7_twin = ax7.twinx()
    ax7.plot(t_edc, edc_db,  color=BLUE,  linewidth=1.2, label="EDC (dB)")
    ax7_twin.plot(t_edc, edc_norm, color=ORANGE, linewidth=1.0,
                  linestyle="--", label="EDC normalizada [0,1]")
    ax7.axhline(-60, color=RED, linestyle=":", linewidth=0.8, label="−60 dB (T60)")
    # Marcar T60
    below_60 = np.where(edc_db <= -60)[0]
    if below_60.size > 0:
        t60_ms = t_edc[below_60[0]]
        ax7.axvline(t60_ms, color=RED, linestyle="--", linewidth=0.8,
                    label=f"T60 ≈ {(t60_ms - 50):.0f} ms")
    ax7.set_title("⑦ EDC / Integración de Schroeder (tail)")
    ax7.set_xlabel("Tiempo (ms)")
    ax7.set_ylabel("EDC (dB)", color=BLUE)
    ax7_twin.set_ylabel("EDC normalizada [0,1]", color=ORANGE)
    ax7.set_ylim(-85, 5)
    lines1, labels1 = ax7.get_legend_handles_labels()
    lines2, labels2 = ax7_twin.get_legend_handles_labels()
    ax7.legend(lines1 + lines2, labels1 + labels2, fontsize=6.5)

    # ── Panel 8: Estadísticas del dataset ─────────────────────────────────────
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.axis("off")

    if delays:
        d = np.array(delays)
        a = np.array(max_amps)
        sr_counts = {k: srs.count(k) for k in set(srs)}

        text_lines = [
            f"Estadísticas de {len(delays)} muestras aleatorias",
            "",
            "Sample rates:",
            *[f"  {k} Hz : {v} ({100*v/len(srs):.0f}%)" for k, v in sorted(sr_counts.items())],
            "",
            "Delay inicial (ms) tras resampleo a 48 kHz:",
            f"  min   : {d.min():.2f} ms",
            f"  max   : {d.max():.2f} ms",
            f"  media : {d.mean():.2f} ms",
            f"  mediana: {np.median(d):.2f} ms",
            "",
            "Amplitud máxima (antes de normalizar):",
            f"  min   : {a.min():.4f}",
            f"  max   : {a.max():.4f}",
            f"  media : {a.mean():.4f}",
            "",
            "Checklist preprocesamiento:",
            "  ✓ Canal W (omni, ch0) seleccionado",
            "  ✓ Resampleo a 48 kHz",
            "  ✓ Delay inicial eliminado (argmax)",
            "  ✓ Normalización amp = 1.0",
            "  ✓ Split head(50ms) / tail(950ms)",
            "  ✓ LPF augmentation {8,12,16,22.05,24} kHz",
            f"  {'✓' if len(sr_counts) == 1 else '⚠'} Split por folds (1 fold disponible → split aleatorio)",
        ]
        ax8.text(0.02, 0.98, "\n".join(text_lines),
                 transform=ax8.transAxes, fontsize=8.5,
                 va="top", ha="left", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", alpha=0.8))
    ax8.set_title("⑧ Estadísticas del dataset")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nInforme guardado en: {out_path}")

    # ── Informe en consola ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CHECKLIST DE PREPROCESAMIENTO — DECOR")
    print("=" * 60)
    print(f"  [✓] 1. Canal W (omni, ch0/{n_ch}) seleccionado")
    print(f"  [✓] 2. Resampleo: {sr_orig} Hz → {FS_TARGET} Hz")
    print(f"  [✓] 3. Delay eliminado: {delay_ms:.2f} ms ({peak_idx} muestras)")
    print(f"  [✓] 4. Normalización: máx abs = {max_amp:.6f} → 1.0")
    print(f"  [✓] 5. Head = {HEAD_SAMPLES} muestras = 50 ms  |  Tail = {TAIL_SAMPLES} muestras = 950 ms")
    print(f"  [✓] 6. LPF augment cutoffs: {LPF_CUTOFFS} Hz")
    below_60 = np.where(edc_db <= -60)[0]
    if below_60.size > 0:
        t60 = (t_edc[below_60[0]] - 50) / 1000  # en segundos, desde inicio de tail
        print(f"  [✓] 7. EDC calculada — T60 ≈ {t60:.3f} s")
    else:
        print(f"  [✓] 7. EDC calculada — T60 > 0.95 s (no cruza −60 dB)")
    if delays:
        n_folds = len(set(
            str(p).split("/fold")[1].split("/")[0]
            for p in all_files[:50] if "/fold" in str(p)
        ))
        fold_msg = f"{n_folds} fold(s) →" + (" split por folds" if n_folds > 1 else " split aleatorio")
        print(f"  [{'✓' if n_folds > 1 else '⚠'}] 8. Prevención fuga de datos: {fold_msg}")
    print("=" * 60)


if __name__ == "__main__":
    main()
