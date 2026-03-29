# DECOR: Deep Room Impulse Response Completion

Este proyecto implementa DECOR (Lin et al., 2025), una red neuronal para predecir la curva de decaimiento de energía (EDC) de la reverberación tardía a partir de los primeros 50 ms (head) de una respuesta impulsional (RIR).

## Estado del proyecto (29/03/2026)

### Completado
- ✅ **Encoder fully convolutional** (`DecorEncoder`): 9 bloques Conv1d + BatchNorm + PReLU, AdaptiveAvgPool1d + proyección 1×1 al espacio latente. Acepta señales de longitud variable.
- ✅ **Decoder con upsampling progresivo** (`DecorDecoder`): 6 etapas de upsampling con bloques residuales (Conv1d → BatchNorm → LeakyReLU), activación Sigmoid y recorte/padding dinámico a longitud objetivo.
- ✅ **Función de pérdida combinada** (`DecorLoss`): L1 temporal + Multi-Resolution STFT (ventanas 512/1024/2048), pesos configurables α/β.
- ✅ **Dataset BIRD** (`BirdDataset`): lazy loading de .wav/.flac/.flaac, auto-discovery de folds, integración de Schroeder para EDC, padding/truncado fijo para batching.
- ✅ **Pipeline de entrenamiento** (`train.py`): Adam, validación por época, guardado del mejor checkpoint.
- ✅ **Evaluación con las 5 métricas del paper** (`eval.py`): MSTFT, EDF MAE/RMSE (dB), T60 MAPE (%), DRR MSE (dB).
- ✅ **Auditoría de datos** (`audit_data_hash.py`): hashing SHA256, verificación de integridad.

### Resultados preliminares (2 épocas, CPU debug)

Entrenamiento sobre 10.000 RIRs del dataset BIRD (fold001), batch_size=4, lr=1e-4:

| Métrica | Paper DECOR | Nuestro (2 épocas) | Observación |
|---|---|---|---|
| MSTFT (↓) | 0.97 | 1.47 | Mismo orden de magnitud |
| EDF MAE (dB, ↓) | 4.04 | 14.91 | Necesita más épocas |
| EDF RMSE (dB, ↓) | 6.19 | 29.91 | Pendiente no capturada aún |
| T60 MAPE (%, ↓) | 14.6 | 100.0 | EDC predicha aún plana |
| DRR MSE (dB, ↓) | 1.15 | 699.83 | Requiere entrenamiento serio |

> **Nota:** Estos resultados son solo una prueba de integración del sistema (2 épocas en CPU). El modelo apenas ha empezado a aprender. Se requiere entrenamiento en GPU (20+ épocas) para evaluar calidad real.

### Pendiente
- ⏳ Entrenamiento completo en GPU (20+ épocas, batch_size=32).
- ⏳ Visualización de EDC predicha vs real.
- ⏳ Learning rate scheduling (ReduceLROnPlateau).
- ⏳ Ajuste de pesos α/β de la loss (MSTFT domina actualmente).
- ⏳ Síntesis de cola temporal desde la EDC predicha.

## Estructura del proyecto

```
models/
  encoder.py         DecorEncoder — fully convolutional, latent_dim=128
  decoder.py         DecorDecoder — upsampling progresivo + bloques residuales
  loss.py            DecorLoss — L1 + Multi-Resolution STFT
scripts/
  bird_loader.py     BirdDataset — lazy loading + Schroeder integration
  dataset.py         RIRDataset — loader .wav con preprocesado clásico
  train.py           Entrenamiento DECOR con validación y checkpoint
  eval.py            Evaluación con 5 métricas del paper
  audit_data_hash.py Auditoría de integridad del dataset
  test_gpu.py        Verificación de entorno GPU
  verify_data.py     Visualización de preprocesado
synthesis/
  reconstruct_rir.py Reconstrucción de RIR (en desarrollo)
```

## Dataset

- **BIRD**: 10.000 archivos .flac en `data/BIRD/Bird/fold001/`
- Frecuencia de muestreo: 48 kHz
- Split cabeza/cola: 50 ms (2.400 muestras)
- Input: head (1, 2400) → Output: EDC normalizada [0,1] (1, 48000)

## Requisitos

- Python 3.10+
- PyTorch + torchaudio
- numpy, matplotlib, tqdm, pyroomacoustics

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Activar entorno
source .venv/bin/activate

# Verificar GPU
python3 scripts/test_gpu.py

# Sanity-check del dataset
python3 scripts/bird_loader.py --root-dir data/BIRD --folds auto

# Entrenar (CPU debug)
python3 scripts/train.py --batch-size 4 --epochs 2 --num-workers 0

# Entrenar (GPU)
python3 scripts/train.py --batch-size 32 --epochs 20

# Evaluar
python3 scripts/eval.py --checkpoint checkpoint.pth --data-root data/BIRD

# Auditar integridad del dataset
python3 scripts/audit_data_hash.py --data-root data --bird-root data/BIRD --manifest data_hash_manifest.sha256
python3 scripts/audit_data_hash.py --verify --manifest data_hash_manifest.sha256
```
