import argparse
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from scripts.bird_loader import BirdDataset
from models.encoder import DecorEncoder
from models.decoder import DecorDecoder
from models.loss import DecorLoss


def build_folds(num_folds: int = 89):
	return [f"fold{i:03d}" for i in range(1, num_folds + 1)]


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


def run_validation(encoder, decoder, criterion, val_loader, device):
	encoder.eval()
	decoder.eval()
	total_val_loss = 0.0

	with torch.no_grad():
		for batch in val_loader:
			head = batch["input"].to(device)
			edc_target = batch["target"].to(device)

			z = encoder(head)
			edc_pred = decoder(z, target_length=edc_target.shape[-1])
			loss_dict = criterion(edc_pred, edc_target)
			total_val_loss += loss_dict["loss"].item()

	return total_val_loss / max(1, len(val_loader))


def main():
	parser = argparse.ArgumentParser(description="Entrenamiento DECOR sobre dataset BIRD")
	parser.add_argument("--data-root", type=str, default="data/BIRD", help="Ruta raíz del dataset BIRD.")
	parser.add_argument("--epochs", type=int, default=20, help="Número de épocas de entrenamiento.")
	parser.add_argument("--batch-size", type=int, default=32, help="Tamaño de batch.")
	parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate para Adam.")
	parser.add_argument("--latent-dim", type=int, default=128, help="Dimensión del espacio latente.")
	parser.add_argument("--num-workers", type=int, default=0, help="Número de workers para DataLoader.")
	parser.add_argument("--val-ratio", type=float, default=0.1, help="Proporción para validación.")
	parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
	parser.add_argument("--checkpoint", type=str, default="checkpoint.pth", help="Ruta para guardar el mejor modelo.")
	args = parser.parse_args()

	random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Dispositivo: {device}")

	dataset = BirdDataset(root_dir=args.data_root, folds=None)

	train_subset, val_subset = build_train_val_subsets(dataset, val_ratio=args.val_ratio, seed=args.seed)

	train_loader = DataLoader(
		train_subset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=(device.type == "cuda"),
	)

	val_loader = None
	if val_subset is not None and len(val_subset) > 0:
		val_loader = DataLoader(
			val_subset,
			batch_size=args.batch_size,
			shuffle=False,
			num_workers=args.num_workers,
			pin_memory=(device.type == "cuda"),
		)

	encoder = DecorEncoder(latent_dim=args.latent_dim).to(device)
	decoder = DecorDecoder(in_channels=args.latent_dim, target_length=48000).to(device)
	criterion = DecorLoss(alpha=1.0, beta=1.0).to(device)
	optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)

	best_train_loss = float("inf")

	for epoch in range(1, args.epochs + 1):
		encoder.train()
		decoder.train()

		running_loss = 0.0
		progress_bar = tqdm(train_loader, desc=f"Época {epoch}/{args.epochs}", leave=True)

		for batch in progress_bar:
			head = batch["input"].to(device)
			edc_target = batch["target"].to(device)

			optimizer.zero_grad()

			z = encoder(head)
			edc_pred = decoder(z, target_length=edc_target.shape[-1])
			loss_dict = criterion(edc_pred, edc_target)
			loss = loss_dict["loss"]

			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			progress_bar.set_postfix(
				loss=f"{loss.item():.5f}",
				l1=f"{loss_dict['l1_loss'].item():.5f}",
				mrstft=f"{loss_dict['mrstft_loss'].item():.5f}",
			)

		epoch_train_loss = running_loss / max(1, len(train_loader))
		print(f"[Epoch {epoch}] Train Loss: {epoch_train_loss:.6f}")

		if epoch_train_loss < best_train_loss:
			best_train_loss = epoch_train_loss
			torch.save(
				{
					"epoch": epoch,
					"encoder_state_dict": encoder.state_dict(),
					"decoder_state_dict": decoder.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"best_train_loss": best_train_loss,
					"config": vars(args),
				},
				args.checkpoint,
			)
			print(f"Nuevo mejor modelo guardado en {args.checkpoint} (loss={best_train_loss:.6f})")

		if val_loader is not None:
			val_loss = run_validation(encoder, decoder, criterion, val_loader, device)
			print(f"[Epoch {epoch}] Val Loss: {val_loss:.6f}")


if __name__ == "__main__":
	main()

