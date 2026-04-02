"""
Script de depuración: 1 época de entrenamiento en CPU con un subconjunto pequeño
del dataset BIRD. Imprime diagnósticos detallados por batch y genera una gráfica
comparando la cola RIR reconstruida vs. la real.

Uso:
    python scripts/debug_train_one_epoch.py [--num-samples 20] [--batch-size 4]
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.bird_loader import BirdDataset
from scripts.train import batch_schroeder_integration
from models.encoder import DecorEncoder
from models.decoder import DecorDecoder
from models.loss import DecorLoss
from pytorch_optimizer import Ranger21


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de diagnóstico
# ─────────────────────────────────────────────────────────────────────────────

def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total ** 0.5


def _print_separator(char: str = "─", width: int = 70) -> None:
    print(char * width)


def _print_model_summary(encoder, decoder) -> None:
    _print_separator("═")
    print("RESUMEN DEL MODELO")
    _print_separator("═")
    enc_params = _count_params(encoder)
    dec_params = _count_params(decoder)
    print(f"  Encoder parámetros entrenables : {enc_params:>12,}")
    print(f"  Decoder parámetros entrenables : {dec_params:>12,}")
    print(f"  TOTAL                          : {enc_params + dec_params:>12,}")
    _print_separator()

    # Encoder
    print("\n  ENCODER")
    print(f"    Bloques           : {len(encoder.encoder_stack)}")
    first_block = encoder.encoder_stack[0]
    k = first_block.main[0].kernel_size[0]
    print(f"    Kernel size       : {k}")
    rf = 1
    stride_acc = 1
    for _ in range(len(encoder.encoder_stack)):
        rf += (k - 1) * stride_acc
        stride_acc *= 2
    print(f"    Campo receptivo   : {rf:,} timesteps ({rf/48000:.2f} s a 48 kHz)")

    # Decoder
    print("\n  DECODER")
    print(f"    num_bands         : {decoder.num_bands}")
    print(f"    num_decays        : {decoder.num_decays}")
    print(f"    filterbank kernel : {decoder.filterbank.kernel_size[0]}")
    print(f"    target_length     : {decoder.target_length}")
    print(f"    out_act           : {decoder.out_act}")
    _print_separator()


def _print_batch_debug(step: int, batch_size: int, z, tail_pred, tail_target,
                       loss_dict, edc_l1, encoder, decoder,
                       t_forward: float, t_backward: float) -> None:
    """Imprime diagnóstico detallado de un batch."""
    _print_separator("─")
    print(f"  BATCH {step:03d}  |  B={batch_size}")
    _print_separator("─")
    print(f"    z          : shape={tuple(z.shape)}  "
          f"mean={z.mean().item():.4f}  std={z.std().item():.4f}")
    print(f"    tail_pred  : shape={tuple(tail_pred.shape)}  "
          f"min={tail_pred.min().item():.4f}  max={tail_pred.max().item():.4f}  "
          f"std={tail_pred.std().item():.5f}")
    print(f"    tail_target: shape={tuple(tail_target.shape)}  "
          f"min={tail_target.min().item():.4f}  max={tail_target.max().item():.4f}")
    print(f"    loss_total : {loss_dict['loss'].item():.6f}")
    print(f"    l1_time    : {loss_dict['l1_loss'].item():.6f}")
    print(f"    mrstft     : {loss_dict['mrstft_loss'].item():.6f}")
    print(f"    EDC-L1     : {edc_l1:.6f}")
    print(f"    grad_norm  : enc={_grad_norm(encoder):.4f}  dec={_grad_norm(decoder):.4f}")
    print(f"    tiempo     : forward={t_forward*1000:.1f} ms  "
          f"backward={t_backward*1000:.1f} ms")


def _check_grad_health(encoder, decoder) -> None:
    all_params = list(encoder.named_parameters()) + list(decoder.named_parameters())
    nan_grads = [(n, p.shape) for n, p in all_params
                 if p.grad is not None and torch.isnan(p.grad).any()]
    inf_grads = [(n, p.shape) for n, p in all_params
                 if p.grad is not None and torch.isinf(p.grad).any()]
    if nan_grads:
        print(f"  ⚠ GRADIENTES NaN en: {[n for n, _ in nan_grads]}")
    if inf_grads:
        print(f"  ⚠ GRADIENTES Inf en: {[n for n, _ in inf_grads]}")
    if not nan_grads and not inf_grads:
        print("  Gradientes: OK (sin NaN ni Inf)")


# ─────────────────────────────────────────────────────────────────────────────
# Gráfica
# ─────────────────────────────────────────────────────────────────────────────

def _save_rir_plot(encoder, decoder, dataset, device: torch.device,
                  sample_idx: int, out_path: str, sample_rate: int = 48000) -> None:
    """Genera gráfica con: RIR head, cola real, cola predicha y EDC comparada."""
    encoder.eval()
    decoder.eval()

    sample = dataset[sample_idx]
    head = sample["input"].unsqueeze(0).to(device)
    tail_target = sample["target"].unsqueeze(0).to(device)
    edc_target = sample["target_edc"].unsqueeze(0).to(device)

    with torch.no_grad():
        z = encoder(head)
        tail_pred = decoder(z, target_length=tail_target.shape[-1])
        edc_pred = batch_schroeder_integration(tail_pred)

    head_np = head[0, 0].cpu().numpy()
    tail_real_np = tail_target[0, 0].cpu().numpy()
    tail_pred_np = tail_pred[0, 0].cpu().numpy()
    edc_real_np = edc_target[0, 0].cpu().numpy()
    edc_pred_np = edc_pred[0, 0].cpu().numpy()

    t_head = [i / sample_rate * 1000 for i in range(len(head_np))]
    t_tail = [50.0 + i / sample_rate * 1000 for i in range(len(tail_real_np))]
    t_edc = [50.0 + i / sample_rate * 1000 for i in range(len(edc_real_np))]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f"DECOR — Muestra #{sample_idx} (1 época de debug)", fontsize=13)

    # Panel 1: head RIR
    axes[0].plot(t_head, head_np, color="steelblue", linewidth=0.6)
    axes[0].set_title("RIR Head (entrada al encoder, 0–50 ms)")
    axes[0].set_xlabel("Tiempo (ms)")
    axes[0].set_ylabel("Amplitud")
    axes[0].axvline(50, color="red", linestyle="--", linewidth=0.8, label="Corte head/tail")
    axes[0].legend(fontsize=8)

    # Panel 2: cola real vs. cola predicha
    axes[1].plot(t_tail, tail_real_np, color="steelblue", linewidth=0.5,
                 alpha=0.7, label="Cola real")
    axes[1].plot(t_tail, tail_pred_np, color="tomato", linewidth=0.8,
                 alpha=0.85, label="Cola predicha")
    axes[1].set_title("Cola RIR: real vs. predicha")
    axes[1].set_xlabel("Tiempo (ms)")
    axes[1].set_ylabel("Amplitud")
    axes[1].legend(fontsize=8)

    # Panel 3: EDC real vs. EDC predicha
    axes[2].plot(t_edc, edc_real_np, color="steelblue", linewidth=1.2, label="EDC real")
    axes[2].plot(t_edc, edc_pred_np, color="tomato", linewidth=1.2,
                 linestyle="--", label="EDC predicha")
    axes[2].set_title("EDC (Schroeder): real vs. predicha")
    axes[2].set_xlabel("Tiempo (ms)")
    axes[2].set_ylabel("EDC normalizada [0,1]")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\n  Gráfica guardada en: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Debug: 1 época DECOR en CPU")
    parser.add_argument("--data-root", type=str, default="data/BIRD")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Número de muestras del dataset a usar (subset pequeño).")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size para el debug en CPU (recomendado 2-8).")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--loss-alpha", type=float, default=0.05)
    parser.add_argument("--loss-beta", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-out", type=str, default="scripts/debug_rir_plot.png",
                        help="Ruta donde guardar la gráfica RIR.")
    parser.add_argument("--plot-sample", type=int, default=0,
                        help="Índice de muestra (dentro del subset) a visualizar.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    # ── Dataset ──────────────────────────────────────────────────────────────
    _print_separator("═")
    print("CARGA DE DATOS")
    _print_separator("═")
    dataset = BirdDataset(root_dir=args.data_root, folds=None)
    indices = list(range(min(args.num_samples, len(dataset))))
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=args.batch_size,
                        shuffle=True, num_workers=0, drop_last=True)

    print(f"  Dataset total : {len(dataset):,} muestras")
    print(f"  Subset debug  : {len(subset)} muestras")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  Num batches   : {len(loader)}")

    sample0 = dataset[0]
    print(f"  input shape   : {tuple(sample0['input'].shape)}")
    print(f"  target shape  : {tuple(sample0['target'].shape)}")
    print(f"  target_edc    : {tuple(sample0['target_edc'].shape)}")

    # ── Modelos ───────────────────────────────────────────────────────────────
    encoder = DecorEncoder(latent_dim=args.latent_dim).to(device)
    decoder = DecorDecoder(in_channels=args.latent_dim, target_length=45600).to(device)
    criterion = DecorLoss(alpha=args.loss_alpha, beta=args.loss_beta).to(device)
    optimizer = Ranger21(
        list(encoder.parameters()) + list(decoder.parameters()),
        num_iterations=max(1, len(loader)),
        lr=args.lr,
    )

    _print_model_summary(encoder, decoder)

    # ── Alineación paper ──────────────────────────────────────────────────────
    _print_separator("═")
    print("VERIFICACIÓN ALINEACIÓN PAPER")
    _print_separator("═")
    print(f"  LR objetivo paper    : 1e-4   →  actual: {args.lr:.1e}  "
          + ("✓" if abs(args.lr - 1e-4) < 1e-10 else f"⚠ ({args.lr:.1e})"))
    print(f"  Batch objetivo paper : 128    →  este debug: {args.batch_size}  "
          + ("✓ (para producción)" if args.batch_size == 128 else
             "(subset debug; producción debe ser 128)"))
    print(f"  Optimizador          : Ranger21  ✓")
    print(f"  MSTFT resoluciones   : 4  ✓")
    print(f"  Ventanas STFT        : [64, 512, 2048, 8192]  ✓")
    print(f"  Hops STFT            : [32, 256, 1024, 4096]  ✓")
    print(f"  Target tail (decoder): T=45600 (950 ms a 48 kHz)  ✓")
    print(f"  n empieza en 0.05 s  : ✓")
    print(f"  out_act decoder      : {decoder.out_act}  ✓")
    print(f"  Filterbank init      : FIR octave-band  ✓")
    _print_separator()

    # ── Entrenamiento 1 época ─────────────────────────────────────────────────
    _print_separator("═")
    print("ENTRENAMIENTO — 1 ÉPOCA DEBUG")
    _print_separator("═")

    encoder.train()
    decoder.train()

    epoch_t0 = time.perf_counter()
    running_loss = 0.0
    running_edc_l1 = 0.0

    for step, batch in enumerate(loader, start=1):
        head = batch["input"].to(device)
        tail_target = batch["target"].to(device)
        edc_target = batch["target_edc"].to(device)

        optimizer.zero_grad()

        t0 = time.perf_counter()
        z = encoder(head)
        tail_pred = decoder(z, target_length=tail_target.shape[-1])
        loss_dict = criterion(tail_pred, tail_target)
        t_forward = time.perf_counter() - t0

        t1 = time.perf_counter()
        loss_dict["loss"].backward()
        optimizer.step()
        t_backward = time.perf_counter() - t1

        with torch.no_grad():
            edc_pred = batch_schroeder_integration(tail_pred)
            edc_l1 = torch.mean(torch.abs(edc_pred - edc_target)).item()

        running_loss += loss_dict["loss"].item()
        running_edc_l1 += edc_l1

        _print_batch_debug(
            step=step,
            batch_size=head.size(0),
            z=z.detach(),
            tail_pred=tail_pred.detach(),
            tail_target=tail_target,
            loss_dict={k: v.detach() for k, v in loss_dict.items()},
            edc_l1=edc_l1,
            encoder=encoder,
            decoder=decoder,
            t_forward=t_forward,
            t_backward=t_backward,
        )
        _check_grad_health(encoder, decoder)

    epoch_time = time.perf_counter() - epoch_t0
    n_batches = max(1, len(loader))

    _print_separator("═")
    print("RESUMEN ÉPOCA")
    _print_separator("═")
    print(f"  Tiempo total          : {epoch_time:.2f} s")
    print(f"  Tiempo medio / batch  : {epoch_time/n_batches*1000:.1f} ms")
    print(f"  Train Loss media      : {running_loss/n_batches:.6f}")
    print(f"  Train EDC-L1 media    : {running_edc_l1/n_batches:.6f}")
    print(f"  Batches procesados    : {n_batches}")
    _print_separator()

    # ── Gráfica ───────────────────────────────────────────────────────────────
    plot_sample_idx = min(args.plot_sample, len(subset) - 1)
    _save_rir_plot(
        encoder=encoder,
        decoder=decoder,
        dataset=dataset,
        device=device,
        sample_idx=indices[plot_sample_idx],
        out_path=args.plot_out,
    )

    _print_separator("═")
    print("FIN DEL DEBUG")
    _print_separator("═")


if __name__ == "__main__":
    main()
