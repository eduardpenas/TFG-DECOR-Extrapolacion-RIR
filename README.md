# DECOR: Deep Room Impulse Response Completion

Implementación de DECOR para extrapolar la reverberación tardía de una RIR a partir de su parte temprana (head de 50 ms).

## Estado del proyecto (30/05/2026)

### Implementado
- Encoder DECOR en [models/encoder.py](models/encoder.py): 14 bloques Conv1d + MLP a espacio latente.
- Decoder acústico en [models/decoder.py](models/decoder.py): envolventes de decaimiento, filtrado por bandas y mezcla final.
- Loss multi-resolución STFT en [models/loss.py](models/loss.py), alineada con resolución temporal/frecuencial del paper.
- Entrenamiento principal en [scripts/train.py](scripts/train.py): split por folds, Ranger21, gradient clipping, validación por época y checkpoint del mejor modelo.
- Evaluación en [scripts/eval.py](scripts/eval.py): métricas MSTFT, EDF MAE/RMSE, T60 MAPE y DRR MSE.
- Utilidades de depuración/inspección:
  - [scripts/debug_train_one_epoch.py](scripts/debug_train_one_epoch.py)
  - [scripts/verify_data.py](scripts/verify_data.py)
  - [scripts/verify_preprocessing.py](scripts/verify_preprocessing.py)
  - [scripts/plot_training_logs.py](scripts/plot_training_logs.py)
  - [scripts/test_sr_robustness.py](scripts/test_sr_robustness.py)
  - [scripts/inspect_audio_files.py](scripts/inspect_audio_files.py)

### Artefactos actualmente presentes en el repo
- Checkpoint de entrenamiento: [checkpoint.pth](checkpoint.pth)
- Log de entrenamiento: [logs_entrenamiento_completo.txt](logs_entrenamiento_completo.txt)
- Notebook de trabajo: [Entrenamiento.ipynb](Entrenamiento.ipynb)
- Manifest de integridad de datos: [data_hash_manifest.sha256](data_hash_manifest.sha256)

### Pendiente / en evolución
- Consolidar resultados finales de evaluación sobre el conjunto objetivo.
- Documentar benchmarking comparativo de manera sistemática en una sección de resultados reproducibles.

## Flujo de datos (importante)

En este repositorio conviven dos rutas de datos:

1. Pipeline sintético (.npy)
- Generación: [scripts/generate_synthetic_rir_dataset.py](scripts/generate_synthetic_rir_dataset.py)
- Validación: [scripts/validate_generated_dataset.py](scripts/validate_generated_dataset.py)
- Estructura esperada en data/raw con subcarpetas head, tail, rirs y edc_tail.

2. Pipeline BIRD (.wav/.flac)
- Dataset usado en entrenamiento/evaluación actual: [scripts/bird_loader.py](scripts/bird_loader.py)
- Carga por folds y separación head/tail para DECOR.

Nota: [scripts/dataset.py](scripts/dataset.py) no es el cargador principal del flujo BIRD actual.

## Arquitectura resumida

```text
head RIR (50 ms) -> DecorEncoder -> z (latent)
z -> DecorDecoder -> tail RIR predicha
tail real/predicha -> losses + métricas (MSTFT, EDC, T60, DRR)
```

## Estructura del proyecto

```text
models/
  encoder.py
  decoder.py
  loss.py
  transformer.py

scripts/
  bird_loader.py
  train.py
  eval.py
  debug_train_one_epoch.py
  generate_synthetic_rir_dataset.py
  validate_generated_dataset.py
  verify_data.py
  verify_preprocessing.py
  inspect_audio_files.py
  plot_training_logs.py
  process_wavs.py
  test_gpu.py
  test_sr_robustness.py
  audit_data_hash.py

synthesis/
  reconstruct_rir.py
```

## Requisitos

- Python 3.10+
- Dependencias en [requirements.txt](requirements.txt)

Instalación:

```bash
pip install -r requirements.txt
```

## Uso rápido

```bash
# Verificar entorno (GPU/torch)
python scripts/test_gpu.py

# Entrenamiento DECOR (BIRD)
python scripts/train.py --data-root data/BIRD --epochs 20 --batch-size 128 --lr 1e-4

# Evaluación con checkpoint
python scripts/eval.py --checkpoint checkpoint.pth --data-root data/BIRD

# Generar/validar dataset sintético
python scripts/generate_synthetic_rir_dataset.py --num-rooms 6000 --output-dir data/raw
python scripts/validate_generated_dataset.py --data-path data/raw --expected-rooms 6000
```

