# DECOR: Deep Room Impulse Response Completion

Este proyecto implementa DECOR (Lin et al., 2025), una red neuronal para predecir la cola de reverberación tardía a partir de los primeros 50 ms (head) de una respuesta impulsional (RIR).

## Estado del proyecto (02/04/2026)

### Completado
- ✅ **Encoder alineado con paper** (`DecorEncoder`): 14 bloques Conv1d (kernel=15, stride=2) + BatchNorm + PReLU, campo receptivo 229.363 timesteps (4,78 s). MLP 3 capas (512→512→256→128) → z∈ℝ¹²⁸.
- ✅ **Decoder físico DECOR** (`DecorDecoder`): base de decaimiento exponencial (N=20 tasas log-espaciadas, T60∈[0,05-3,0]s), MLP 7 capas ocultas con cabezas A′ y C′ (máscara de sparsidad A=A′⊙σ(C′)), banco de filtros FIR octave-band ISO 266 (M=10 bandas, orden 1023), band mixer Conv1d 1×1 sin activación. n empieza en 0,05 s (offset de head).
- ✅ **Función de pérdida MSTFT** (`DecorLoss`): spectral convergence + log-magnitude sobre 4 resoluciones [64, 512, 2048, 8192], hops [32, 256, 1024, 4096], ventana Hann.
- ✅ **Optimizador Ranger21** (`train.py`): `num_iterations = epochs × len(train_loader)`, lr=1e-4.
- ✅ **Gradient clipping** (`train.py`): `clip_grad_norm_(max_norm=1.0)` antes de `optimizer.step()`. Norma antes del clip: ~700; después: 1.0.
- ✅ **Target: cola RIR cruda** (`bird_loader.py`): el modelo predice la cola temporal stocástica; la EDC (Schroeder) se computa como métrica secundaria.
- ✅ **Dataset BIRD** (`BirdDataset`): lazy loading de .wav/.flac, integración de Schroeder para EDC, padding/truncado fijo para batching. Devuelve `input` (head 2400 muestras), `target` (tail 48000 muestras), `target_edc`.
- ✅ **Script de depuración** (`debug_train_one_epoch.py`): 1 época en CPU con subconjunto pequeño, diagnóstico por batch (grad norms, tiempos, NaN/Inf), verificación de alineación con paper y gráfica RIR head+cola real/predicha+EDC.
- ✅ **Pipeline de entrenamiento** (`train.py`): Ranger21, validación por época, guardado del mejor checkpoint, barra de progreso con loss/l1/mrstft/edc_l1/gnorm.

### Resultados debug (1 época, CPU, 20 muestras, batch=4)

| Métrica | Tras correcciones |
|---|---|
| Train Loss (MRSTFT) | ~7.5 |
| EDC-L1 | ~0.55 |
| Grad norm (tras clipping) | 1.0000 ✓ |
| DC offset cola predicha | Eliminado (bias band_mixer = 0) |

> **Nota:** Estos resultados son una prueba de integración (1 época CPU, subset 20 muestras). Para evaluar calidad real se requiere entrenamiento completo en GPU (20+ épocas, batch=128).

### Pendiente
- ⏳ Entrenamiento completo en GPU (20+ épocas, batch_size=128).
- ⏳ Evaluación con las 5 métricas del paper (eval.py).
- ⏳ Learning rate scheduling.

## Arquitectura

```
head RIR (1, 2400)
      │
  DecorEncoder (14 bloques Conv1d, kernel=15, stride=2)
      │  AdaptiveAvgPool1d → MLP 3 capas
      │
  z ∈ ℝ¹²⁸
      │
  DecorDecoder
      ├─ MLP 7 capas → A′, C′  →  A = A′ ⊙ σ(C′)   (B, M, N)
      ├─ decay_basis E(n)  →  envelopes Y = A @ E   (B, M, T)
      ├─ white noise → FIR octave filterbank          (B, M, T)
      └─ shaped_bands = Y ⊙ filtered_noise → band_mixer Conv1d 1×1
      │
  cola RIR predicha (1, 1, 45600)  [950 ms @ 48 kHz]
```

## Estructura del proyecto

```
models/
  encoder.py              DecorEncoder — 14 bloques, kernel=15, MLP 3 capas
  decoder.py              DecorDecoder — físico DECOR, FIR octave-band
  loss.py                 DecorLoss — MSTFT 4 resoluciones
  transformer.py          (reservado)
scripts/
  bird_loader.py          BirdDataset — tail RIR + EDC Schroeder
  train.py                Entrenamiento: Ranger21, grad clipping, checkpoint
  debug_train_one_epoch.py  1 época debug CPU + gráfica RIR
  eval.py                 Evaluación con métricas del paper
  audit_data_hash.py      Auditoría SHA256 del dataset
  test_gpu.py             Verificación de entorno GPU
  verify_data.py          Visualización de preprocesado
synthesis/
  reconstruct_rir.py      Reconstrucción de RIR (en desarrollo)
```

## Dataset

- **BIRD**: 10.000 archivos .flac en `data/BIRD/Bird/fold001/`
- Frecuencia de muestreo: 48 kHz
- Split cabeza/cola: 50 ms (2.400 muestras)
- Input: head (1, 2400) → Target: cola RIR cruda (1, 48000)

## Requisitos

- Python 3.10+
- PyTorch + torchaudio
- numpy, matplotlib, tqdm, scipy, pyroomacoustics, pytorch-optimizer

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Activar entorno
source .venv/bin/activate

# Verificar GPU
python scripts/test_gpu.py

# Debug 1 época en CPU (20 muestras)
python scripts/debug_train_one_epoch.py --data-root data/BIRD --num-samples 20 --batch-size 4

# Entrenar (GPU completo)
python scripts/train.py --data-root data/BIRD --epochs 20 --batch-size 128 --lr 1e-4

# Evaluar
python scripts/eval.py --checkpoint checkpoint.pth --data-root data/BIRD

# Auditar integridad del dataset
python scripts/audit_data_hash.py --data-root data --bird-root data/BIRD --manifest data_hash_manifest.sha256
```

