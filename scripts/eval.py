"""
Evaluación DECOR con las 5 métricas del paper (Lin et al., 2025):
  1. MSTFT (↓)  — Multi-Resolution STFT Loss
  2. EDF MAE (dB, ↓) — Error absoluto medio de la EDC en dB
  3. EDF RMSE (dB, ↓) — Raíz del error cuadrático medio de la EDC en dB
  4. T60 MAPE (%, ↓) — Error porcentual absoluto medio del T60
  5. DRR MSE (dB, ↓) — Error cuadrático medio del Direct-to-Reverberant Ratio

Uso:
  python3 scripts/eval.py --checkpoint checkpoint.pth --data-root data/BIRD
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.bird_loader import BirdDataset, schroeder_integration
from models.encoder import DecorEncoder
from models.decoder import DecorDecoder
from models.loss import MultiResolutionSTFTLoss


# ──────────────────────────────────────────────
# Métricas individuales
# ──────────────────────────────────────────────

def edc_to_db(edc_norm: torch.Tensor, db_floor: float = -80.0) -> torch.Tensor:
    """Convierte EDC normalizada [0,1] de vuelta a escala dB [db_floor, 0]."""
    return edc_norm * (-db_floor) + db_floor  # 0→0 dB, 1→db_floor dB... invertido
    # La normalización original: norm = (edc_db - db_floor) / (0 - db_floor)
    # Inversa: edc_db = norm * (0 - db_floor) + db_floor = -norm * db_floor + db_floor
    # edc_db = db_floor * (1 - norm)


def _edc_norm_to_db(edc_norm: torch.Tensor, db_floor: float = -80.0) -> torch.Tensor:
    """Inversa exacta de la normalización [0,1] → dB."""
    # norm = (edc_db - db_floor) / (0 - db_floor)
    # edc_db = norm * (0 - db_floor) + db_floor = db_floor * (1 - norm)
    return db_floor * (1.0 - edc_norm)


def compute_edf_mae_db(
    pred: torch.Tensor, target: torch.Tensor, db_floor: float = -80.0
) -> torch.Tensor:
    """EDF MAE en dB — media sobre batch y muestras temporales."""
    pred_db = _edc_norm_to_db(pred, db_floor)
    target_db = _edc_norm_to_db(target, db_floor)
    return torch.abs(pred_db - target_db).mean()


def compute_edf_rmse_db(
    pred: torch.Tensor, target: torch.Tensor, db_floor: float = -80.0
) -> torch.Tensor:
    """EDF RMSE en dB — raíz de MSE sobre batch y muestras temporales."""
    pred_db = _edc_norm_to_db(pred, db_floor)
    target_db = _edc_norm_to_db(target, db_floor)
    return torch.sqrt(((pred_db - target_db) ** 2).mean())


def estimate_t60_from_edc_db(edc_db: torch.Tensor, fs: int = 48000) -> torch.Tensor:
    """
    Estima T60 a partir de una EDC en dB de forma (B, 1, L).
    Busca el punto donde la EDC cruza -60 dB por interpolación lineal.
    Si no cruza, extrapola linealmente desde -5 dB a -25 dB (método Schroeder).
    Devuelve tensor de forma (B,) en segundos.
    """
    batch_size = edc_db.size(0)
    length = edc_db.size(-1)
    t60_values = torch.zeros(batch_size, device=edc_db.device)

    for b in range(batch_size):
        curve = edc_db[b, 0]  # (L,)

        # Buscar cruce directo a -60 dB
        below_60 = (curve <= -60.0).nonzero(as_tuple=False)
        if below_60.numel() > 0:
            idx = below_60[0].item()
            t60_values[b] = idx / fs
            continue

        # Extrapolación lineal desde rango -5 dB a -25 dB
        mask_5 = (curve <= -5.0).nonzero(as_tuple=False)
        mask_25 = (curve <= -25.0).nonzero(as_tuple=False)

        if mask_5.numel() > 0 and mask_25.numel() > 0:
            i1 = mask_5[0].item()
            i2 = mask_25[0].item()
            if i2 > i1:
                db_drop = curve[i2] - curve[i1]
                time_drop = (i2 - i1) / fs
                # Extrapolar a -60 dB
                t60_values[b] = time_drop * (-60.0 / db_drop.item())
            else:
                t60_values[b] = length / fs  # fallback
        else:
            t60_values[b] = length / fs  # fallback

    return t60_values


def compute_t60_mape(
    pred: torch.Tensor, target: torch.Tensor, db_floor: float = -80.0, fs: int = 48000, eps: float = 1e-6
) -> torch.Tensor:
    """T60 MAPE (%) — Error porcentual absoluto medio del T60."""
    pred_db = _edc_norm_to_db(pred, db_floor)
    target_db = _edc_norm_to_db(target, db_floor)

    t60_pred = estimate_t60_from_edc_db(pred_db, fs=fs)
    t60_target = estimate_t60_from_edc_db(target_db, fs=fs)

    mape = torch.abs(t60_pred - t60_target) / (torch.abs(t60_target) + eps) * 100.0
    return mape.mean()


def compute_drr_from_edc_db(edc_db: torch.Tensor, head_samples: int = 2400) -> torch.Tensor:
    """
    Estima DRR (dB) a partir de la EDC completa en dB.
    DRR = 10*log10(E_direct / E_reverberant)
    Aproximación: usa el valor de la EDC al inicio vs al punto head_samples.
    Forma de entrada: (B, 1, L). Devuelve (B,).
    """
    # La EDC en dB al sample 0 es ~0 dB (energía total)
    # La EDC en dB al sample head_samples indica la energía restante tras la parte directa
    # DRR ≈ E_total - E_reverb (en lineal) → en dB
    edc_at_0 = edc_db[:, 0, 0]  # ~0 dB
    length = edc_db.size(-1)
    idx = min(head_samples, length - 1)
    edc_at_head = edc_db[:, 0, idx]

    # Energía directa (lineal) = 10^(edc_0/10) - 10^(edc_head/10)
    e_total = 10.0 ** (edc_at_0 / 10.0)
    e_reverb = 10.0 ** (edc_at_head / 10.0)
    e_direct = torch.clamp(e_total - e_reverb, min=1e-12)

    drr_db = 10.0 * torch.log10(e_direct / (e_reverb + 1e-12))
    # Evita que outliers extremos dominen el MSE de DRR.
    drr_db = torch.clamp(drr_db, min=-60.0, max=60.0)
    return drr_db


def compute_drr_mse_db(
    pred: torch.Tensor, target: torch.Tensor, db_floor: float = -80.0, head_samples: int = 2400
) -> torch.Tensor:
    """DRR MSE (dB) — Error cuadrático medio del DRR."""
    pred_db = _edc_norm_to_db(pred, db_floor)
    target_db = _edc_norm_to_db(target, db_floor)

    drr_pred = compute_drr_from_edc_db(pred_db, head_samples=head_samples)
    drr_target = compute_drr_from_edc_db(target_db, head_samples=head_samples)

    return ((drr_pred - drr_target) ** 2).mean()


# ──────────────────────────────────────────────
# Bucle de evaluación
# ──────────────────────────────────────────────

def evaluate(encoder, decoder, dataloader, device, mrstft_criterion):
    encoder.eval()
    decoder.eval()

    accum = {
        "mstft": 0.0,
        "edf_mae_db": 0.0,
        "edf_rmse_sq": 0.0,  # acumular MSE para luego hacer sqrt global
        "t60_mape": 0.0,
        "drr_mse": 0.0,
        "count": 0,
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluando"):
            head = batch["input"].to(device)
            edc_target = batch["target"].to(device)

            z = encoder(head)
            edc_pred = decoder(z, target_length=edc_target.shape[-1])

            # La EDC normalizada del dataset está en [0,1].
            # Acotar predicción/target estabiliza métricas en dB (T60/DRR).
            edc_pred = torch.clamp(edc_pred, 0.0, 1.0)
            edc_target = torch.clamp(edc_target, 0.0, 1.0)

            bs = head.size(0)
            accum["count"] += bs

            # 1. MSTFT
            accum["mstft"] += mrstft_criterion(edc_pred, edc_target).item() * bs

            # 2. EDF MAE (dB)
            accum["edf_mae_db"] += compute_edf_mae_db(edc_pred, edc_target).item() * bs

            # 3. EDF RMSE (dB) — acumular suma de cuadrados
            pred_db = _edc_norm_to_db(edc_pred)
            target_db = _edc_norm_to_db(edc_target)
            accum["edf_rmse_sq"] += ((pred_db - target_db) ** 2).sum().item() / (
                edc_pred.size(-1) * edc_pred.size(1)
            )

            # 4. T60 MAPE (%)
            accum["t60_mape"] += compute_t60_mape(edc_pred, edc_target).item() * bs

            # 5. DRR MSE (dB)
            accum["drr_mse"] += compute_drr_mse_db(edc_pred, edc_target).item() * bs

    n = accum["count"]
    results = {
        "MSTFT": accum["mstft"] / n,
        "EDF_MAE_dB": accum["edf_mae_db"] / n,
        "EDF_RMSE_dB": (accum["edf_rmse_sq"] / n) ** 0.5,
        "T60_MAPE_%": accum["t60_mape"] / n,
        "DRR_MSE_dB": accum["drr_mse"] / n,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluación DECOR (5 métricas del paper)")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth", help="Ruta al checkpoint.")
    parser.add_argument("--data-root", type=str, default="data/BIRD", help="Raíz del dataset BIRD.")
    parser.add_argument("--batch-size", type=int, default=8, help="Tamaño de batch para evaluación.")
    parser.add_argument("--num-workers", type=int, default=0, help="Workers del DataLoader.")
    parser.add_argument("--latent-dim", type=int, default=128, help="Dimensión latente (debe coincidir con el checkpoint).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # Cargar checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: No se encuentra el checkpoint en {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Checkpoint cargado: época {ckpt.get('epoch', '?')}, train loss {ckpt.get('best_train_loss', '?'):.6f}")

    # Modelos
    encoder = DecorEncoder(latent_dim=args.latent_dim).to(device)
    decoder = DecorDecoder(in_channels=args.latent_dim, target_length=48000).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    decoder.load_state_dict(ckpt["decoder_state_dict"])

    # Dataset (auto-discovery)
    dataset = BirdDataset(root_dir=args.data_root, folds=None)
    print(f"Muestras de evaluación: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Criterio MSTFT
    mrstft = MultiResolutionSTFTLoss().to(device)

    # Evaluar
    results = evaluate(encoder, decoder, dataloader, device, mrstft)

    # Resultados
    print("\n" + "=" * 50)
    print("RESULTADOS — Métricas DECOR (Lin et al., 2025)")
    print("=" * 50)
    print(f"  MSTFT        (↓): {results['MSTFT']:.4f}")
    print(f"  EDF MAE  (dB, ↓): {results['EDF_MAE_dB']:.2f}")
    print(f"  EDF RMSE (dB, ↓): {results['EDF_RMSE_dB']:.2f}")
    print(f"  T60 MAPE  (%, ↓): {results['T60_MAPE_%']:.1f}")
    print(f"  DRR MSE  (dB, ↓): {results['DRR_MSE_dB']:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
