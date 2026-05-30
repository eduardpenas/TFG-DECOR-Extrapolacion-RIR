"""
Test de robustez del sistema DECOR ante distintas frecuencias de muestreo.

Verifica que:
  1. BirdDataset resamulea cualquier sr de entrada a 48 kHz y produce siempre
     tensores de forma (1, 2400) para head y (1, 45600) para tail.
  2. La normalización y eliminación de delay inicial funcionan correctamente.
  3. La augmentación LPF atenúa energía por encima del cutoff elegido.
  4. El forward pass Encoder → Decoder produce (B, 1, 45600) sin importar el sr
     original del audio.
  5. Con augment=True el modelo recibe señales con distintos anchos de banda,
     lo que entrena la robustez al sr descrita en el paper (§3.2).

Uso:
    python scripts/test_sr_robustness.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.decoder import DecorDecoder
from models.encoder import DecorEncoder
from scripts.bird_loader import BirdDataset

# ──────────────────────────────────────────────────────────────────────────────
# Constantes del paper
# ──────────────────────────────────────────────────────────────────────────────
FS_TARGET = 48_000
HEAD_SAMPLES = 2_400    # 50 ms × 48 kHz
TAIL_SAMPLES = 45_600   # 950 ms × 48 kHz
SAMPLE_RATES = [8_000, 16_000, 22_050, 44_100, 48_000]
LOWPASS_CUTOFFS_PAPER = (8_000, 12_000, 16_000, 22_050, 24_000)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sep(char="─", w=65):
    print(char * w)


def _ok(cond: bool, msg: str = ""):
    status = "OK  " if cond else "FALL"
    print(f"    [{status}]  {msg}")
    if not cond:
        raise AssertionError(f"Test fallido: {msg}")


def make_synthetic_rir(sr: int, duration_s: float = 1.0, seed: int = 0) -> np.ndarray:
    """RIR sintética: impulso directo + ruido con decaimiento exponencial."""
    rng = np.random.default_rng(seed)
    n = int(sr * duration_s)
    rir = np.zeros(n, dtype=np.float32)
    peak = max(1, n // 20)          # impulso directo a ~5 % de la duración
    rir[peak] = 1.0
    t = np.arange(n - peak) / sr
    rir[peak:] += (rng.standard_normal(n - peak) * np.exp(-6.0 * t)).astype(np.float32)
    return rir


def create_wav_files(tmpdir: str) -> list:
    """Crea un WAV sintético por cada sample rate de SAMPLE_RATES."""
    paths = []
    for sr in SAMPLE_RATES:
        rir = make_synthetic_rir(sr)
        path = Path(tmpdir) / f"rir_{sr}hz.wav"
        sf.write(str(path), rir, sr)
        paths.append(path)
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# TEST 1: Formas de salida independientes del sr de entrada
# ──────────────────────────────────────────────────────────────────────────────

def test_output_shapes():
    _sep("═")
    print("TEST 1 — Formas de salida (input/target/edc) invariantes al sr")
    _sep("═")

    with tempfile.TemporaryDirectory() as tmpdir:
        create_wav_files(tmpdir)
        ds = BirdDataset(root_dir=tmpdir, augment=False)

        for i, sr in enumerate(SAMPLE_RATES):
            sample = ds[i]
            h = tuple(sample["input"].shape)
            t = tuple(sample["target"].shape)
            e = tuple(sample["target_edc"].shape)
            ok = h == (1, HEAD_SAMPLES) and t == (1, TAIL_SAMPLES) and e == (1, TAIL_SAMPLES)
            _ok(ok, f"sr={sr:>6} Hz → input{h}  target{t}  edc{e}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# TEST 2: Normalización y eliminación de delay
# ──────────────────────────────────────────────────────────────────────────────

def test_normalization_and_delay_removal():
    _sep("═")
    print("TEST 2 — Normalización amp=1.0 y eliminación de delay inicial")
    _sep("═")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Crear RIR con delay artificial: silencio en primeros 200 ms
        sr = 48_000
        n = sr  # 1 segundo
        rir = np.zeros(n, dtype=np.float32)
        peak = int(0.2 * sr)      # delay de 200 ms
        rir[peak] = 0.5           # amplitud != 1 → debe normalizarse
        t = np.arange(n - peak) / sr
        rir[peak:] += (np.random.default_rng(1).standard_normal(n - peak)
                       * np.exp(-6.0 * t) * 0.3).astype(np.float32)

        path = Path(tmpdir) / "rir_delayed.wav"
        sf.write(str(path), rir, sr)

        ds = BirdDataset(root_dir=tmpdir, augment=False)
        sample = ds[0]
        head = sample["input"][0]   # (2400,)

        max_abs = head.abs().max().item()
        _ok(abs(max_abs - 1.0) < 1e-4,
            f"Amplitud máxima del head tras normalización = {max_abs:.6f} (esperado ≈1.0)")

        # La primera muestra del head debe ser el impulso (peak tras delay removal)
        first_sample = head[0].item()
        _ok(abs(first_sample - 1.0) < 1e-4,
            f"Primera muestra del head = {first_sample:.6f} (esperado ≈1.0, delay eliminado)")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# TEST 3: Augmentación LPF atenúa energía de alta frecuencia
# ──────────────────────────────────────────────────────────────────────────────

def test_lowpass_augmentation():
    _sep("═")
    print("TEST 3 — Augmentación LPF atenúa energía por encima del cutoff")
    _sep("═")

    import torchaudio

    t = torch.linspace(0, 1.0, FS_TARGET)
    NYQUIST = FS_TARGET // 2

    for cutoff in LOWPASS_CUTOFFS_PAPER:
        # Tono de prueba a 1.5× el cutoff, limitado al 90 % de Nyquist.
        # Solo tiene sentido afirmar atenuación si el tono está *por encima* del cutoff.
        test_freq = min(int(cutoff * 1.5), int(NYQUIST * 0.90))
        signal = torch.sin(2 * 3.14159265 * test_freq * t)

        filtered = torchaudio.functional.lowpass_biquad(
            signal.unsqueeze(0),
            sample_rate=FS_TARGET,
            cutoff_freq=float(cutoff),
        ).squeeze(0)

        energy_in = signal.pow(2).mean().item()
        energy_out = filtered.pow(2).mean().item()
        attenuation_db = 10 * np.log10(max(energy_out, 1e-12) / energy_in)

        if test_freq > cutoff:
            # El tono está por encima del corte → debe atenuarse (≥3 dB)
            atenua = energy_out < energy_in * 0.5
            _ok(atenua,
                f"cutoff={cutoff:>6} Hz  tono={test_freq} Hz  "
                f"atten={attenuation_db:.1f} dB  (tono > cutoff, debe atenuar)")
        else:
            # Cutoff tan alto que no hay rango de frecuencia útil para atenuación
            # (ej. cutoff=24k Hz en 48k → Nyquist; comportamiento esperado: sin efecto)
            print(f"    [SKIP]  cutoff={cutoff:>6} Hz  tono={test_freq} Hz ≤ cutoff "
                  f"→ no atenuación esperada (cutoff ≈ Nyquist)")

    # Verificar que el dataset usa el flag augment correctamente
    with tempfile.TemporaryDirectory() as tmpdir:
        create_wav_files(tmpdir)
        ds_aug = BirdDataset(root_dir=tmpdir, augment=True)
        ds_plain = BirdDataset(root_dir=tmpdir, augment=False)

        # Tomar la muestra a 48 kHz (índice 4): sin resamplear, solo el LPF cambia
        idx_48k = SAMPLE_RATES.index(48_000)
        plain = ds_plain[idx_48k]["input"]

        # Muestrear varias veces con augmentación; al menos una debe diferir
        aug_samples = [ds_aug[idx_48k]["input"] for _ in range(10)]
        some_differ = any(not torch.allclose(plain, s, atol=1e-5) for s in aug_samples)
        _ok(some_differ, "Con augment=True al menos una muestra difiere de la plain")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# TEST 4: Forward pass Encoder → Decoder invariante al sr de entrada
# ──────────────────────────────────────────────────────────────────────────────

def test_full_pipeline():
    _sep("═")
    print("TEST 4 — Forward pass Encoder → Decoder invariante al sr de entrada")
    _sep("═")

    encoder = DecorEncoder(latent_dim=128)
    decoder = DecorDecoder(in_channels=128)
    encoder.eval()
    decoder.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        create_wav_files(tmpdir)
        ds = BirdDataset(root_dir=tmpdir, augment=False)

        with torch.no_grad():
            for i, sr in enumerate(SAMPLE_RATES):
                head = ds[i]["input"].unsqueeze(0)    # (1, 1, 2400)
                z = encoder(head)                     # (1, 128)
                tail_pred = decoder(z)                # (1, 1, 45600)

                ok_z = z.shape == (1, 128)
                ok_tail = tail_pred.shape == (1, 1, TAIL_SAMPLES)
                ok_finite = torch.isfinite(z).all() and torch.isfinite(tail_pred).all()

                _ok(ok_z and ok_tail and ok_finite,
                    f"sr={sr:>6} Hz → head{tuple(head.shape)} "
                    f"→ z{tuple(z.shape)} → tail_pred{tuple(tail_pred.shape)}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# TEST 5: Consistencia del split head/tail en muestras
# ──────────────────────────────────────────────────────────────────────────────

def test_head_tail_split_consistency():
    _sep("═")
    print("TEST 5 — Consistencia temporal head/tail (50 ms / 950 ms a 48 kHz)")
    _sep()

    with tempfile.TemporaryDirectory() as tmpdir:
        create_wav_files(tmpdir)
        ds = BirdDataset(root_dir=tmpdir, augment=False)

        for i, sr in enumerate(SAMPLE_RATES):
            sample = ds[i]
            head_len = sample["input"].shape[-1]
            tail_len = sample["target"].shape[-1]
            head_ms = head_len / FS_TARGET * 1000
            tail_ms = tail_len / FS_TARGET * 1000
            total_ms = head_ms + tail_ms

            ok = (head_len == HEAD_SAMPLES and tail_len == TAIL_SAMPLES
                  and abs(total_ms - 1000.0) < 0.1)
            _ok(ok,
                f"sr={sr:>6} Hz → head={head_len} muestras ({head_ms:.0f} ms)  "
                f"tail={tail_len} muestras ({tail_ms:.0f} ms)  "
                f"total={total_ms:.0f} ms")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# TEST 6: Resampleo correcto — señal a 16 kHz no tiene contenido >8 kHz tras resampleo
# ──────────────────────────────────────────────────────────────────────────────

def test_resampling_bandwidth():
    _sep("═")
    print("TEST 6 — Señal a 16 kHz carece de contenido >8 kHz tras resampleo a 48 kHz")
    _sep()

    with tempfile.TemporaryDirectory() as tmpdir:
        sr_orig = 16_000
        # Señal blanca limitada al ancho de banda de 16 kHz (Nyquist = 8 kHz)
        rng = np.random.default_rng(42)
        rir = rng.standard_normal(sr_orig).astype(np.float32)
        rir[sr_orig // 20] = 1.0   # pico al inicio
        path = Path(tmpdir) / "rir_16k.wav"
        sf.write(str(path), rir, sr_orig)

        ds = BirdDataset(root_dir=tmpdir, augment=False)
        full = torch.cat([ds[0]["input"][0], ds[0]["target"][0]])  # (48000,)

        # STFT para ver energía por encima de 8 kHz
        spec = torch.stft(
            full,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            window=torch.hann_window(1024),
            return_complex=True,
        )
        freqs = torch.linspace(0, FS_TARGET / 2, spec.shape[0])  # Hz por bin
        mask_above_8k = freqs > 8_000
        energy_above = spec[mask_above_8k].abs().pow(2).mean().item()
        energy_total = spec.abs().pow(2).mean().item()
        ratio = energy_above / (energy_total + 1e-12)

        _ok(ratio < 0.05,
            f"Energía >8 kHz / total = {ratio:.5f} (esperado <0.05; rolloff del resampler Kaiser)")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _sep("═")
    print("TEST DE ROBUSTEZ A FRECUENCIA DE MUESTREO — DECOR")
    print(f"  Target sr : {FS_TARGET} Hz")
    print(f"  Head      : {HEAD_SAMPLES} muestras = 50 ms")
    print(f"  Tail      : {TAIL_SAMPLES} muestras = 950 ms")
    print(f"  Cutoffs LPF paper : {LOWPASS_CUTOFFS_PAPER} Hz")
    _sep("═")
    print()

    test_output_shapes()
    test_normalization_and_delay_removal()
    test_lowpass_augmentation()
    test_full_pipeline()
    test_head_tail_split_consistency()
    test_resampling_bandwidth()

    _sep("═")
    print("TODOS LOS TESTS PASADOS")
    _sep("═")
