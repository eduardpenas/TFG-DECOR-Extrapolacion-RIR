# DECOR: Deep Room Impulse Response Completion
Este proyecto implementa DECOR, una red neuronal para sintetizar la reverberación tardía (tail) a partir de los primeros 50 ms (head) de una respuesta impulsional.

## Estado del proyecto (17/02/2026)
- ✅ **Arquitectura base definida**: `DecorEncoder` y `AcousticDecoder` están implementados y listos para integrarse.
- ✅ **Dataset y preprocesado**: carga de .wav, recorte de retardo inicial, normalización y partición head/tail.
- ✅ **Herramientas de verificación**: script de validación de GPU y script para comprobar el preprocesado con visualización.
- ⏳ **Pendiente**: entrenamiento, evaluación cuantitativa, y generación del tail completo a partir del head.

## Estructura actual
- `models/encoder.py`: Encoder convolucional con 9 bloques (reducción temporal por stride=2).
- `models/decoder.py`: Decoder acústico MLP que produce la matriz de amplitudes (bandas × decaimientos).
- `scripts/dataset.py`: Dataset `RIRDataset` con segmentación head (0–50 ms) y tail (50 ms–1 s).
- `scripts/verify_data.py`: Verificación visual del preprocesado y guardado de gráfica.
- `scripts/test_gpu.py`: Comprobación de entorno y disponibilidad de GPU.

## Requisitos
- Python 3.10+
- PyTorch + torchaudio
- Matplotlib

## Uso rápido
### Verificar GPU
Ejecuta el script de entorno para confirmar CUDA.

### Verificar datos
Coloca archivos .wav en `data/` y ejecuta la verificación.
El script genera `scripts/verificacion_data.png`.

## Próximos pasos (plan mínimo)
1. Implementar el pipeline de entrenamiento end-to-end (head → encoder → decoder → síntesis tail).
2. Definir la función de pérdida (por ejemplo, $L_1$ en dominio temporal y/o espectral).
3. Crear scripts de entrenamiento y evaluación (métricas y ejemplos de reconstrucción).
4. Documentar el conjunto de datos y el protocolo experimental.
