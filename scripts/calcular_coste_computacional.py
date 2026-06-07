import argparse
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from models.decoder import DecorDecoder
from models.encoder import DecorEncoder


def count_trainable_parameters(model: torch.nn.Module) -> int:
	return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def estimate_parameter_memory_mb(total_parameters: int, bytes_per_parameter: int = 4) -> float:
	return (total_parameters * bytes_per_parameter) / (1024 ** 2)


def synchronize(device: torch.device) -> None:
	if device.type == "cuda":
		torch.cuda.synchronize(device)


def benchmark_pipeline(
	encoder: torch.nn.Module,
	decoder: torch.nn.Module,
	device: torch.device,
	head_length: int,
	target_length: int,
	sample_rate: int,
	warmup_iterations: int,
	benchmark_iterations: int,
) -> dict[str, float]:
	encoder.eval()
	decoder.eval()

	dummy_head = torch.randn(1, 1, head_length, device=device)

	with torch.inference_mode():
		for _ in range(warmup_iterations):
			latent = encoder(dummy_head)
			_ = decoder(latent, target_length=target_length)

		synchronize(device)
		start_time = time.perf_counter()
		for _ in range(benchmark_iterations):
			latent = encoder(dummy_head)
			_ = decoder(latent, target_length=target_length)
		synchronize(device)
		end_time = time.perf_counter()

	avg_inference_time = (end_time - start_time) / max(1, benchmark_iterations)
	audio_duration = target_length / float(sample_rate)
	rtf = avg_inference_time / audio_duration

	return {
		"avg_inference_time_s": avg_inference_time,
		"audio_duration_s": audio_duration,
		"rtf": rtf,
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Calcula el coste computacional del pipeline DECOR (encoder + decoder)."
	)
	parser.add_argument("--latent-dim", type=int, default=128, help="Dimensión del espacio latente.")
	parser.add_argument("--head-length", type=int, default=2400, help="Longitud de la head de entrada en muestras.")
	parser.add_argument("--target-length", type=int, default=45600, help="Longitud de la cola a generar en muestras.")
	parser.add_argument("--sample-rate", type=int, default=48000, help="Frecuencia de muestreo en Hz.")
	parser.add_argument("--warmup-iterations", type=int, default=10, help="Iteraciones de calentamiento antes de medir.")
	parser.add_argument("--iterations", type=int, default=100, help="Iteraciones para la medición final.")
	parser.add_argument(
		"--decoder-activation",
		type=str,
		default="none",
		choices=("none", "sigmoid", "tanh"),
		help="Activación de salida del decoder.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("--- Análisis de Coste Computacional DECOR ---")
	print(f"Dispositivo: {device}")

	if device.type == "cuda":
		torch.backends.cudnn.benchmark = True
		print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

	encoder = DecorEncoder(latent_dim=args.latent_dim).to(device)
	decoder = DecorDecoder(
		in_channels=args.latent_dim,
		target_length=args.target_length,
		output_activation=args.decoder_activation,
		sample_rate=args.sample_rate,
	).to(device)

	params_enc = count_trainable_parameters(encoder)
	params_dec = count_trainable_parameters(decoder)
	total_params = params_enc + params_dec
	memory_mb = estimate_parameter_memory_mb(total_params)

	benchmark = benchmark_pipeline(
		encoder=encoder,
		decoder=decoder,
		device=device,
		head_length=args.head_length,
		target_length=args.target_length,
		sample_rate=args.sample_rate,
		warmup_iterations=args.warmup_iterations,
		benchmark_iterations=args.iterations,
	)

	print(f"Parámetros encoder: {params_enc:,}")
	print(f"Parámetros decoder: {params_dec:,}")
	print(f"Parámetros totales: {total_params:,}")
	print(f"Huella estimada de memoria (float32): {memory_mb:.2f} MB")
	print(f"Tiempo medio de inferencia: {benchmark['avg_inference_time_s'] * 1000:.2f} ms")
	print(f"Duración de salida: {benchmark['audio_duration_s']:.4f} s")
	print(f"RTF (Real-Time Factor): {benchmark['rtf']:.4f}")


if __name__ == "__main__":
	main()