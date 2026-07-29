# Evaluación de Codificadores de Video para la Detección de Anomalías en Tiempo Real

**Tesis para optar al título de Ingeniero Civil en Informática**
Universidad de Santiago de Chile — Facultad de Ingeniería
Departamento de Ingeniería Informática

**Autor:** Diego Valdés
**Año:** 2025

---

## Descripción

Este repositorio contiene el código fuente completo del sistema propuesto para comparar codificadores de video modernos en la tarea de detección de anomalías (VAD) sobre el dataset UCF-Crime. Se evalúan cuatro codificadores bajo tres configuraciones de entrenamiento cada uno:

- **Baseline:** codificador congelado + clasificador MLP
- **LoRA + Clasificador:** codificador adaptado con LoRA + MLP (entrenamiento conjunto)
- **Solo LoRA:** codificador adaptado con LoRA + MLP congelado del Baseline

Los codificadores evaluados son:

| Codificador | Checkpoint | Frames internos | Embedding |
|---|---|---|---|
| I3D | `rgb_imagenet.pt` | 32 | 1024 |
| TimeSformer | `facebook/timesformer-base-finetuned-k400` | 8 | 768 |
| X-CLIP | `microsoft/xclip-base-patch32` | 8 | 512 |
| Video Swin | `Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1` | 32 | 1024 |

---

## Estructura del repositorio

```
tesis-vad-encoders/
│
├── README.md
├── requirements.txt
├── .gitignore
│
└── src/
    ├── Preprocesamiento/
    │   ├── eda.py                          # Análisis exploratorio sobre la población completa
    │   └── preprocess.py                   # Muestreo balanceado y generación de index_clips.csv
    │
    ├── entrenamientoF/                     # Scripts de entrenamiento final + resultados por codificador
    │   ├── i3d_final_completo.py
    │   ├── timesformer_final_r16a16d024 (1).py
    │   ├── xclip_final_r16a16d024.py
    │   ├── swin_final_r16a16d024 (1).py
    │   ├── i3d_02_curva_roc.png
    │   ├── i3d_final_r16a16d024_resultados.json
    │   ├── resultados clip/                # Métricas y figuras de X-CLIP
    │   ├── resultados swin/                # Métricas y figuras de Video Swin
    │   └── resultados times/                # Métricas y figuras de TimeSformer
    │
    └── evaluacion/
        ├── medir_operacional_i3d.py         # FPS, ms/clip, GFLOPs — I3D
        ├── medir_operacional_ts.py          # FPS, ms/clip, GFLOPs — TimeSformer
        ├── medir_operacional_xclip.py       # FPS, ms/clip, GFLOPs — X-CLIP
        └── medir_operacional_swin.py        # FPS, ms/clip, GFLOPs — Video Swin
```

> Los resultados de I3D (curva ROC y JSON de métricas) quedaron directamente
> dentro de `entrenamientoF/`, mientras que los de TimeSformer, X-CLIP y
> Video Swin están en sus respectivas subcarpetas `resultados times/`,
> `resultados clip/` y `resultados swin/`.

---

## Requisitos

Python 3.10+ y CUDA 11.8+. Instalar dependencias:

```bash
pip install -r requirements.txt
```

Contenido de `requirements.txt`:

```
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.38.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
opencv-python>=4.8.0
thop>=0.1.1
fvcore>=0.1.5
```

---

## Datos

El experimento utiliza un subconjunto de **UCF-Crime** con 700 videos de 6 categorías (5 anómalas + Normal). El dataset original está disponible en:

> Sultani, W., Chen, C., & Shah, M. (2018). Real-World Anomaly Detection in Surveillance Videos. CVPR.
> https://www.crcv.ucf.edu/projects/real-world/

Una vez descargados los videos, generar el índice de clips:

```bash
python3 src/Preprocesamiento/preprocess.py
# Genera processed/index_clips.csv, requerido antes de entrenar
```

---

## Uso

### 1. Entrenamiento

Cada script es independiente y reanudable (usa checkpoints de fase):

```bash
# I3D — incluye grid search del clasificador y calibración LoRA
screen -S i3d
python3 "src/entrenamientoF/i3d_final_completo.py" 2>&1 | tee i3d.log

# TimeSformer (~82-97 h por configuración LoRA)
screen -S ts
python3 "src/entrenamientoF/timesformer_final_r16a16d024 (1).py" 2>&1 | tee ts.log

# X-CLIP (~37-39 h por configuración LoRA)
screen -S xclip
python3 "src/entrenamientoF/xclip_final_r16a16d024.py" 2>&1 | tee xclip.log

# Video Swin Transformer
screen -S swin
python3 "src/entrenamientoF/swin_final_r16a16d024 (1).py" 2>&1 | tee swin.log
```

### 2. Métricas operacionales

Una vez generados los checkpoints, medir latencia y GFLOPs:

```bash
python3 src/evaluacion/medir_operacional_i3d.py
python3 src/evaluacion/medir_operacional_ts.py
python3 src/evaluacion/medir_operacional_xclip.py
python3 src/evaluacion/medir_operacional_swin.py
```

Cada script actualiza el JSON de resultados correspondiente con una sección `"operational"` por configuración.

### 3. Resultados

Las métricas de desempeño, curvas ROC, matrices de confusión y figuras comparativas (frontera de Pareto, gráfico de burbujas AUC–GFLOPs) quedan disponibles dentro de `src/entrenamientoF/` en las subcarpetas correspondientes a cada codificador.

---

## Configuración experimental

Todos los scripts comparten la misma configuración de MLP y LoRA, calibrada sobre I3D:

| Hiperparámetro | Valor |
|---|---|
| Arquitectura MLP | FC (hidden=128, dropout=0.3) |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0.24 |
| Optimizador | Adam |
| LR Baseline/Solo LoRA | 1e-3 |
| LR LoRA+MLP | 1e-4 |
| Épocas máximas | 50 |
| Early stopping | Paciencia 8 (AUC val) |
| Batch size entrenamiento | 64 |
| Batch size métricas operacionales | 1 |

---

## Resultados principales

Mejor configuración por codificador sobre el conjunto de prueba:

| Codificador | Config | AUC | FAR | ms/clip | GFLOPs |
|---|---|---|---|---|---|
| I3D | LoRA+MLP | 0.9149 | 0.1293 | 18.12 | 55.69 |
| TimeSformer | **Solo LoRA** | **0.9413** | 0.1136 | 51.18 | 197.91 |
| X-CLIP | LoRA+MLP | 0.9242 | 0.1874 | **18.42** | **36.11** |
| Video Swin | LoRA+MLP | 0.9295 | 0.1465 | 148.26 | 289.89 |

Umbral de tiempo real: ≤ 533 ms/clip (30 FPS, stride 16 frames).
**Los cuatro codificadores operan dentro del umbral**, con un factor de holgura mínimo de 3.6× (Video Swin).

**Configuración seleccionada: TimeSformer + Solo LoRA**, por combinar el mayor AUC (0.9413) y la menor FAR (0.1136) del conjunto, con holgura de tiempo real de 10.4×.

Se confirma además que LoRA supera al Baseline de codificador congelado en los cuatro codificadores evaluados, aunque la variante óptima (LoRA+MLP vs. Solo LoRA) depende de cada codificador: en I3D, X-CLIP y Video Swin es LoRA+MLP; TimeSformer es la única excepción, donde Solo LoRA resulta óptima porque su Baseline es el más fuerte del conjunto y su adaptación realinea las representaciones de forma favorable a la frontera de decisión heredada.

---

## Licencia

MIT License — libre para reutilizar, modificar y distribuir con atribución.

---

## Citación

Si utilizas este código en tu investigación, por favor cita:

```
Valdés, D. (2025). Evaluación de codificadores de video para la
detección de anomalías en tiempo real. Tesis de Ingeniería Civil
en Informática, Universidad de Santiago de Chile.
```
