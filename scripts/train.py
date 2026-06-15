import argparse
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pytorch_optimizer import Ranger21


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from scripts.synthetic_loader import SyntheticRIRDataset
from scripts.synthetic_loader import schroeder_integration
from scripts.bird_loader import BirdDataset
from models.encoder import DecorEncoder
from models.decoder import DecorDecoder
from models.loss import DecorLoss


def build_train_val_subsets(dataset, val_ratio: float, seed: int):
	num_samples = len(dataset)
	indices = list(range(num_samples))

	rng = random.Random(seed)
	rng.shuffle(indices)

	if num_samples < 2 or val_ratio <= 0.0:
		return Subset(dataset, indices), None

	val_size = max(1, int(num_samples * val_ratio))
	train_size = num_samples - val_size
	if train_size <= 0:
		return Subset(dataset, indices), None

	train_indices = indices[:train_size]
	val_indices = indices[train_size:]
	return Subset(dataset, train_indices), Subset(dataset, val_indices)


def batch_schroeder_integration(tails: torch.Tensor) -> torch.Tensor:
	if tails.dim() != 3 or tails.size(1) != 1:
		raise ValueError("tails debe tener forma (Batch, 1, Length).")

	edc_batch = []
	for i in range(tails.size(0)):
		edc = schroeder_integration(tails[i, 0])
		edc_batch.append(edc)

	return torch.stack(edc_batch, dim=0).unsqueeze(1)


def parse_folds_arg(folds_arg: str):
	if not folds_arg or folds_arg.strip().lower() in {"auto", "none", ""}:
		return None

	parsed = []
	for token in folds_arg.split(","):
		token = token.strip()
		if not token:
			continue
		parsed.append(int(token) if token.isdigit() else token)

	return parsed or None


def run_validation(encoder, decoder, criterion, val_loader, device):
	encoder.eval()
	decoder.eval()
	total_val_loss = 0.0
	total_val_edc_l1 = 0.0

	with torch.no_grad():
		for batch in val_loader:
			head = batch["input"].to(device)
			tail_target = batch["target"].to(device)
			edc_target = batch["target_edc"].to(device)

			z = encoder(head)
			tail_pred = decoder(z, target_length=tail_target.shape[-1])
			loss_dict = criterion(tail_pred, tail_target)
			total_val_loss += loss_dict["loss"].item()

			edc_pred = batch_schroeder_integration(tail_pred)
			total_val_edc_l1 += torch.mean(torch.abs(edc_pred - edc_target)).item()

	mean_val_loss = total_val_loss / max(1, len(val_loader))
	mean_val_edc_l1 = total_val_edc_l1 / max(1, len(val_loader))
	return mean_val_loss, mean_val_edc_l1


def main():
	parser = argparse.ArgumentParser(description="Entrenamiento DECOR sobre datasets synthetic (.npy) o BIRD (.wav/.flac)")
	parser.add_argument(
		"--dataset-type",
		type=str,
		default="synthetic",
		choices=["synthetic", "bird"],
		help="Tipo de dataset: 'synthetic' (.npy + metadata.csv) o 'bird' (audios).",
	)
	parser.add_argument(
		"--data-root",
		type=str,
		default="data/raw_train_sintetico_v1",
		help="Ruta raíz del dataset seleccionado.",
	)
	parser.add_argument(
		"--folds",
		type=str,
		default="auto",
		help="Solo para BIRD: folds separados por comas (ej: fold001,2,3) o 'auto'.",
	)
	parser.add_argument(
		"--bird-augment",
		action="store_true",
		help="Solo para BIRD: activa augmentación con lowpass aleatorio.",
	)
	parser.add_argument(
		"--bird-channel-select",
		type=str,
		default="omni",
		choices=["omni", "random", "mean"],
		help="Solo para BIRD: selección de canal al convertir a mono.",
	)
	parser.add_argument("--epochs", type=int, default=20, help="Número de épocas de entrenamiento.")
	parser.add_argument("--batch-size", type=int, default=32, help="Tamaño de batch.")
	parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate para Ranger21.")
	parser.add_argument("--latent-dim", type=int, default=128, help="Dimensión del espacio latente.")
	parser.add_argument("--loss-alpha", type=float, default=0.0, help="Peso de L1 temporal en la loss total.")
	parser.add_argument("--loss-beta", type=float, default=1.0, help="Peso de MSTFT en la loss total.")
	parser.add_argument("--num-workers", type=int, default=0, help="Número de workers para DataLoader.")
	parser.add_argument("--val-ratio", type=float, default=0.1, help="Proporción para validación.")
	parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
	parser.add_argument("--checkpoint", type=str, default="checkpoint.pth", help="Ruta base para guardar checkpoints (se generan best_ y current_).")  
	parser.add_argument("--resume", type=str, default=None, help="Ruta de un checkpoint para reanudar el entrenamiento.")
	parser.add_argument("--grad-clip", type=float, default=1.0, help="Norma máxima para gradient clipping (0 = desactivado).")
	parser.add_argument("--force-cpu", action="store_true", help="Fuerza entrenamiento en CPU")
	parser.add_argument(
		"--max-samples",
		type=int,
		default=None,
		help="Limita muestras del dataset para pruebas rápidas (None = usar todas)",
	)
	args = parser.parse_args()

	best_checkpoint_path = Path(args.checkpoint)
	best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

	current_checkpoint_path = best_checkpoint_path.parent / f"current_{best_checkpoint_path.name}"

	random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)

	device = torch.device("cpu" if args.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
	print(f"Dispositivo: {device}")

	if device.type == "cuda":
		torch.backends.cudnn.benchmark = True

	if args.dataset_type == "synthetic":
		dataset_base = SyntheticRIRDataset(root_dir=args.data_root, max_samples=args.max_samples)
	else:
		folds = parse_folds_arg(args.folds)
		dataset_base = BirdDataset(
			root_dir=args.data_root,
			folds=folds,
			augment=args.bird_augment,
			channel_select=args.bird_channel_select,
		)
		if args.max_samples is not None and args.max_samples > 0:
			n_limited = min(args.max_samples, len(dataset_base))
			dataset_base = Subset(dataset_base, list(range(n_limited)))

	train_subset, val_subset = build_train_val_subsets(dataset_base, val_ratio=args.val_ratio, seed=args.seed)
	print(
		f"Split {args.dataset_type} aleatorio: "
		f"train={len(train_subset)} | val={len(val_subset) if val_subset is not None else 0}"
	)

	train_loader = DataLoader(
		train_subset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=(device.type == "cuda"),
		persistent_workers=(args.num_workers > 0),
	)

	val_loader = None
	if val_subset is not None and len(val_subset) > 0:
		val_loader = DataLoader(
			val_subset,
			batch_size=args.batch_size,
			shuffle=False,
			num_workers=args.num_workers,
			pin_memory=(device.type == "cuda"),
			persistent_workers=(args.num_workers > 0),
		)

	encoder = DecorEncoder(latent_dim=args.latent_dim).to(device)
	decoder = DecorDecoder(in_channels=args.latent_dim, target_length=45600).to(device)
	criterion = DecorLoss(alpha=args.loss_alpha, beta=args.loss_beta).to(device)
	num_iterations = max(1, args.epochs * len(train_loader))
	optimizer = Ranger21(
		list(encoder.parameters()) + list(decoder.parameters()),
		num_iterations=num_iterations,
		lr=args.lr,
	)
	scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

	start_epoch = 1
	best_val_loss = float("inf")

	if args.resume is not None:
		ckpt_path = Path(args.resume)
		if not ckpt_path.is_file():
			raise FileNotFoundError(f"Checkpoint no encontrado: {ckpt_path}")
		print(f"Reanudando desde {ckpt_path} ...")
		ckpt = torch.load(ckpt_path, map_location=device)
		encoder.load_state_dict(ckpt["encoder_state_dict"])
		decoder.load_state_dict(ckpt["decoder_state_dict"])
		optimizer.load_state_dict(ckpt["optimizer_state_dict"])
		if "scaler_state_dict" in ckpt:
			scaler.load_state_dict(ckpt["scaler_state_dict"])
		start_epoch = ckpt["epoch"] + 1
		best_val_loss = ckpt.get("best_val_loss", ckpt.get("best_train_loss", float("inf")))
		print(f"  Reanudando desde época {start_epoch}/{args.epochs}  |  mejor val_loss conocida: {best_val_loss:.6f}")

	all_params = list(encoder.parameters()) + list(decoder.parameters())

	for epoch in range(start_epoch, args.epochs + 1):
		encoder.train()
		decoder.train()

		running_loss = 0.0
		running_edc_l1 = 0.0
		progress_bar = tqdm(train_loader, desc=f"Época {epoch}/{args.epochs}", leave=True)

		for batch in progress_bar:
			head = batch["input"].to(device)
			tail_target = batch["target"].to(device)
			edc_target = batch["target_edc"].to(device)

			optimizer.zero_grad(set_to_none=True)

			with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
				z = encoder(head)
				tail_pred = decoder(z, target_length=tail_target.shape[-1])
				loss_dict = criterion(tail_pred, tail_target)
				loss = loss_dict["loss"]

			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			grad_norm = torch.nn.utils.clip_grad_norm_(
				all_params,
				max_norm=args.grad_clip if args.grad_clip > 0 else float("inf"),
			).item()
			scaler.step(optimizer)
			scaler.update()

			with torch.no_grad():
				edc_pred = batch_schroeder_integration(tail_pred)
				edc_l1 = torch.mean(torch.abs(edc_pred - edc_target))

			running_loss += loss.item()
			running_edc_l1 += edc_l1.item()
			progress_bar.set_postfix(
				loss=f"{loss.item():.5f}",
				l1=f"{loss_dict['l1_loss'].item():.5f}",
				mrstft=f"{loss_dict['mrstft_loss'].item():.5f}",
				edc_l1=f"{edc_l1.item():.5f}",
				gnorm=f"{grad_norm:.2f}",
			)

		epoch_train_loss = running_loss / max(1, len(train_loader))
		epoch_train_edc_l1 = running_edc_l1 / max(1, len(train_loader))
		print(f"[Epoch {epoch}] Train Loss: {epoch_train_loss:.6f} | Train EDC-L1: {epoch_train_edc_l1:.6f}")

		if val_loader is not None:
			val_loss, val_edc_l1 = run_validation(encoder, decoder, criterion, val_loader, device)
			print(f"[Epoch {epoch}] Val Loss: {val_loss:.6f} | Val EDC-L1: {val_edc_l1:.6f}")
			checkpoint_loss = val_loss
		else:
			val_loss = None
			checkpoint_loss = epoch_train_loss

		checkpoint_payload = {
			"epoch": epoch,
			"encoder_state_dict": encoder.state_dict(),
			"decoder_state_dict": decoder.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"scaler_state_dict": scaler.state_dict(),
			"best_val_loss": best_val_loss,
			"train_loss": epoch_train_loss,
			"val_loss": val_loss,
			"config": vars(args),
		}

		torch.save(checkpoint_payload, current_checkpoint_path)
		print(f"Checkpoint actual guardado en {current_checkpoint_path}")

		if checkpoint_loss < best_val_loss:
			best_val_loss = checkpoint_loss
			criterio = "val" if val_loader is not None else "train"
			checkpoint_payload["best_val_loss"] = best_val_loss
			torch.save(checkpoint_payload, best_checkpoint_path)
			print(f"Nuevo mejor modelo guardado en {best_checkpoint_path} (mejor {criterio}_loss={best_val_loss:.6f})")


if __name__ == "__main__":
	main()

