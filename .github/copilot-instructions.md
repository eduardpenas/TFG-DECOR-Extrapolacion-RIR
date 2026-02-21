# Copilot Instructions for DECOR (TFG-DECOR-Extrapolacion-RIR)

## Project scope and current state
- This repo focuses on DECOR: predict/generate late reverberation from early RIR content.
- Current implemented pieces are: synthetic-data generation/validation and core model blocks.
- Training/evaluation pipeline is not implemented yet; avoid inventing non-existing trainers unless requested.

## Big-picture architecture
- `scripts/generate_synthetic_rir_dataset.py` creates synthetic shoebox RIR data with `pyroomacoustics`.
- Generated sample artifacts are split into:
  - `rirs/sample_XXXXX.npy` (full normalized RIR)
  - `head/sample_XXXXX.npy` (first 50 ms)
  - `tail/sample_XXXXX.npy` (after 50 ms)
  - `edc_tail/sample_XXXXX.npy` (Schroeder EDC over tail)
- Per-sample metadata is stored in `data/raw/metadata.csv`; retry failures go to `generation_errors.csv`.
- `scripts/validate_generated_dataset.py` is the integrity gate for generated `.npy` datasets.
- `scripts/verify_data.py` provides quick human inspection and saves `scripts/verificacion_data.png`.
- Model blocks live in `models/encoder.py` (`DecorEncoder`) and `models/decoder.py` (`AcousticDecoder`).

## Important data-flow nuance
- There are two dataset paths in this repo:
  - Synthetic pipeline (`.npy` + `metadata.csv`) in `data/raw/*`.
  - `scripts/dataset.py` (`RIRDataset`) that loads `.wav` files directly from a folder.
- Do not assume `RIRDataset` currently reads the synthetic `.npy` format.

## Project conventions (specific to this codebase)
- Default sampling rate is `48000` Hz across scripts.
- Head/tail split is fixed to `50 ms` (`2400` samples at 48 kHz).
- Arrays are typically saved as `float32`; computations may use `float64` for stability first.
- Generation is resilient: each sample retries up to `--max-retries-per-room` before counting as skipped.
- Validation checks expected physical ranges (room dims, absorption, min source-receiver distance, RT60 sanity).
- CLI/help text and logs are in Spanish; keep language consistent when modifying these scripts.

## Developer workflows (run from repo root)
- Environment check:
  - `python scripts/test_gpu.py`
- Generate synthetic dataset:
  - `python scripts/generate_synthetic_rir_dataset.py --num-rooms 6000 --output-dir data/raw`
- Validate generated dataset:
  - `python scripts/validate_generated_dataset.py --data-path data/raw --expected-rooms 6000`
- Visualize one sample:
  - `python scripts/verify_data.py --data-path data/raw --sample-id 0`

## Editing guidance for AI agents
- Prefer extending existing scripts rather than creating new frameworks.
- Keep changes surgical and compatible with current file layout under `data/raw/`.
- When changing generation fields or paths, update both generator and validator in the same task.
- If you alter head/tail definitions (`fs`, `head_ms`), propagate changes consistently across generation, validation, and visualization.