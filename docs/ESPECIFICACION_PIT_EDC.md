# Especificación técnica — Physics-Informed Transformer-EDC (PIT-EDC)

Fecha: 2026-02-21  
Proyecto: DECOR (Extrapolación de RIR)

## 1) Objetivo
Definir una arquitectura entrenable y físicamente consistente para extrapolar la reverberación tardía (tail) de una RIR a partir del tramo temprano (head), priorizando la predicción de descriptores energéticos (EDC multibanda) frente a la predicción directa de forma de onda.

## 2) Motivación
La predicción de waveform muestra alta varianza y ruido fase-dependiente. Para mejorar estabilidad y generalización:
- se predice primero la dinámica de energía (EDC),
- se imponen restricciones físicas suaves en la pérdida,
- y se sintetiza la señal de tail en una etapa posterior y controlable.

## 3) Alcance en este repositorio
Esta especificación se alinea con el estado actual del repo:
- Dataset sintético `.npy` disponible en `data/raw/{head,tail,edc_tail,rirs}`.
- `DecorEncoder` implementado en `models/encoder.py`.
- `AcousticDecoder` implementado en `models/decoder.py` (salida de matriz de amplitudes).
- No existe todavía un pipeline de entrenamiento/evaluación end-to-end.

### Fuera de alcance (versión inicial)
- Entrenamiento sobre datasets reales con dominio mixto.
- PINN “duro” resolviendo PDE completa de onda en malla espacial.
- Optimización perceptual avanzada (p. ej. métricas binaurales).

## 4) Arquitectura propuesta

## 4.1 Entrada
Por muestra:
- `head`: primeros 50 ms de la RIR (a 48 kHz, 2400 muestras).
- `metadata`: parámetros físicos de sala (dimensiones, absorción/material equivalente, distancia fuente-receptor, etc.).

Formato sugerido:
- `head`: tensor `(B, 1, 2400)`.
- `metadata`: tensor `(B, Dm)` normalizado.

## 4.2 Encoder de características
Base: `DecorEncoder` (CNN 1D por bloques stride=2).
- Salida latente: `z_audio ∈ R^(B, Dz)`.
- Proyección de metadatos: MLP corto `metadata -> z_meta ∈ R^(B, Dz_meta)`.
- Fusión: concatenación + proyección lineal:

`z0 = Linear([z_audio || z_meta])`

## 4.3 Embedding tipo Koopman (inspirado en Geneva)
Objetivo: mapear el estado latente a un espacio donde la evolución temporal sea aproximadamente lineal.

Bloque recomendado (versión práctica):
- `phi = MLP(z0)`
- transición latente discreta compartida: `phi_{t+1} = A * phi_t + b`
- `A` parametrizada con regularización espectral para estabilidad.

Nota: en MVP, este bloque puede activarse como “residual opcional” para facilitar ablation (`on/off`).

## 4.4 Núcleo predictor temporal
Decoder autoregresivo causal con Transformer (masked self-attention):
- Entrada: token inicial + condición global (`z0` o `phi0`).
- Salida por paso: vector EDC por banda.

Salida objetivo:
- `edc_pred`: `(B, T_tail, Bf)` o `(B, Bf, T_tail)` según convención.
- `Bf`: nº de bandas (ej. 8–16).
- `T_tail`: nº de pasos temporales de la cola (submuestreado o resolución plena).

Recomendación MVP:
- usar resolución temporal reducida para EDC (más estable y eficiente) y luego reinterpolar para síntesis.

## 4.5 Decoder acústico / síntesis de tail
Dos etapas desacopladas:
1. Predicción EDC multibanda (`edc_pred`).
2. Síntesis de `tail` con ruido blanco filtrado moldeado por EDC.

Polaridad/fase:
- aplicar esquema Random Sign-Sticky (RSS) para evitar incoherencias de signo entre muestras adyacentes, especialmente en bajas frecuencias.

## 5) Función de pérdida (física informada, versión suave)
Pérdida total:

`L = λ_edc * L_edc + λ_mono * L_mono + λ_rt60 * L_rt60 + λ_smooth * L_smooth (+ λ_tail * L_tail opcional)`

Componentes:
- `L_edc`: error principal entre `edc_pred` y `edc_gt` (L1 o Huber).
- `L_mono`: penaliza incrementos locales positivos en EDC (debe decrecer globalmente).
- `L_rt60`: alinea pendiente de decaimiento estimada con RT60 esperado/estimado.
- `L_smooth`: regulariza oscilaciones no físicas entre pasos y/o bandas.
- `L_tail` (opcional): pérdida temporal o espectral sobre la cola sintetizada.

Importante:
- Empezar sin PDE acústica completa.
- Introducir restricciones físicas gradualmente para evitar inestabilidad.

## 6) Datos, preprocesado y particiones

## 6.1 Convenciones obligatorias
- Frecuencia de muestreo: 48 kHz.
- Corte head/tail: 50 ms (`2400` muestras para head).
- Tipado de almacenamiento: `float32`.

## 6.2 Split sugerido
- Train/Val/Test por sala (no por muestra aleatoria) para evitar fuga de información.
- Guardar seeds y lista de IDs por split para reproducibilidad.

## 6.3 Normalización
- `head`: normalización por pico o RMS (consistente con generador).
- `metadata`: estandarización por columna (`μ, σ` de train).
- `edc`: trabajar en dominio log-energía cuando sea estable numéricamente.

## 7) Métricas de evaluación
Primarias:
- MAE/RMSE sobre EDC multibanda.
- Error en RT60 estimado por banda y global.
- Coherencia de pendiente de decaimiento (linealidad en dB).

Secundarias:
- Error espectral de la cola sintetizada.
- Correlación temporal entre `tail_pred` y `tail_gt`.
- Métrica de monotonicidad violada (% de frames con incremento espurio).

## 8) Plan de implementación por fases

Fase 0 — Baseline entrenable:
1. Dataset unificado para `.npy` (`head`, `edc_tail`, `tail`, `metadata.csv`).
2. Modelo `Encoder + Transformer causal + cabeza EDC`.
3. Pérdida `L_edc + L_mono`.
4. Script de entrenamiento mínimo con checkpoints y validación.

Fase 1 — Física suave:
1. Añadir `L_rt60` y `L_smooth`.
2. Ajustar pesos `λ` con barrido corto.
3. Registrar curvas de aprendizaje y errores por banda.

Fase 2 — Koopman + síntesis:
1. Activar bloque Koopman opcional (ablation obligatoria).
2. Integrar síntesis tail por ruido filtrado + RSS.
3. Evaluar impacto en métricas temporales/espectrales.

Fase 3 — Consolidación:
1. Selección de mejor configuración por validación.
2. Informe de resultados con tablas y ejemplos de reconstrucción.
3. Preparación de integración en memoria (`memoria/`).

## 9) Criterios de aceptación (MVP)
Se considera MVP válido cuando:
- entrena de forma estable sin divergencia,
- reduce `L_edc` frente a baseline trivial,
- mantiene monotonicidad de EDC en la mayoría de muestras,
- mejora error de RT60 respecto a baseline simple.

## 10) Riesgos y mitigaciones
- Riesgo: sobrecomplejidad temprana (Koopman + PINN + síntesis a la vez).  
  Mitigación: activar por etapas y mantener ablations sistemáticas.

- Riesgo: sobreajuste a metadatos sintéticos.  
  Mitigación: split por sala, regularización y pruebas sin metadatos.

- Riesgo: inestabilidad de pérdidas físicas fuertes.  
  Mitigación: comenzar con restricciones suaves y pesos bajos.

## 11) Decisiones técnicas recomendadas
- Framework: PyTorch (continuidad con el repo).
- Logging: pérdidas por término + métricas por banda.
- Reproducibilidad: seeds fijas, guardado de config en cada experimento.
- Complejidad inicial: priorizar modelo funcional y medible antes de sofisticación física completa.

## 12) Entregables esperados
- Dataset loader unificado para pipeline sintético `.npy`.
- Script de entrenamiento MVP.
- Script de evaluación con métricas EDC/RT60.
- Figura(s) de comparación `edc_gt vs edc_pred` y `tail_gt vs tail_pred`.
- Actualización de README con instrucciones de ejecución.

---

## Resumen ejecutivo
La arquitectura PIT-EDC es técnicamente adecuada para este proyecto y ofrece un equilibrio entre interpretabilidad física y capacidad de modelado temporal. La estrategia recomendada es iterativa: primero un baseline robusto de predicción EDC, luego añadir restricciones físicas suaves y, finalmente, integrar Koopman y síntesis completa de cola con RSS.
