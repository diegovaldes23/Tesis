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
├── src/
│   ├── entrenamiento/
│   │   ├── train_i3d.py              # I3D: grid search + calibración LoRA + 3 configs
│   │   ├── train_timesformer.py      # TimeSformer: 3 configuraciones
│   │   ├── train_xclip.py            # X-CLIP: 3 configuraciones
│   │   └── train_videoswin.py        # Video Swin: 3 configuraciones
│   │
│   └── evaluacion/
│       ├── medir_operacional_i3d.py      # FPS, ms/clip, GFLOPs — I3D
│       ├── medir_operacional_ts.py       # FPS, ms/clip, GFLOPs — TimeSformer
│       ├── medir_operacional_xclip.py    # FPS, ms/clip, GFLOPs — X-CLIP
│       └── medir_operacional_swin.py     # FPS, ms/clip, GFLOPs — Video Swin
│
├── processed/                        # Ignorado por git (ver .gitignore)
│   ├── results/                      # Salidas I3D (.pth, .json, figuras)
│   ├── ts_results/                   # Salidas TimeSformer
│   ├── xclip_results/                # Salidas X-CLIP
│   └── realswin_results/             # Salidas Video Swin
│
└── notebooks/
    └── analisis_comparativo.ipynb    # Pareto, scoring compuesto, figuras
```

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
# El índice processed/index_clips.csv debe existir antes de entrenar
# Generado por el pipeline de preprocesamiento descrito en la Sección 5.1 de la tesis
```

---

## Uso

### 1. Entrenamiento

Cada script es independiente y reanudable (usa checkpoints de fase):

```bash
# I3D — incluye grid search del clasificador y calibración LoRA
screen -S i3d
python3 src/entrenamiento/train_i3d.py 2>&1 | tee i3d.log

# TimeSformer (~82-97 h por configuración LoRA)
screen -S ts
python3 src/entrenamiento/train_timesformer.py 2>&1 | tee ts.log

# X-CLIP (~37-39 h por configuración LoRA)
screen -S xclip
python3 src/entrenamiento/train_xclip.py 2>&1 | tee xclip.log

# Video Swin Transformer
screen -S swin
python3 src/entrenamiento/train_videoswin.py 2>&1 | tee swin.log
```

Para retomar desde una fase específica sin re-ejecutar todo:

```bash
# Ejemplo: re-entrenar Solo LoRA de TimeSformer
rm processed/ts_results/.phase_solo_lora_done
python3 src/entrenamiento/train_timesformer.py
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

### 3. Análisis comparativo

Abrir el notebook para reproducir el análisis de Pareto, scoring compuesto y figuras de la tesis:

```bash
jupyter notebook notebooks/analisis_comparativo.ipynb
```

---

## Configuración experimental

Todos los scripts comparten la misma configuración de MLP y LoRA, calibrada sobre I3D:

| Hiperparámetro | Valor |
|---|---|
| Arquitectura MLP | FC (hidden=128, dropout=0.3) |
| LoRA rank | 4 |
| LoRA alpha | 8 |
| LoRA dropout | 0.10 |
| Optimizador | Adam |
| LR Baseline/Solo LoRA | 1e-3 |
| LR LoRA+MLP | 1e-4 |
| Épocas máximas | 50 |
| Early stopping | Paciencia 8 (AUC val) |
| Batch size entrenamiento | 64 |
| Batch size métricas operacionales | 1 |

---

## Resultados principales

| Codificador | Config | AUC | FAR | ms/clip | GFLOPs |
|---|---|---|---|---|---|
| I3D | LoRA+MLP | 0.871 | 0.155 | 43.1 | 55.8 |
| TimeSformer | Solo LoRA | **0.941** | 0.135 | 160.7 | 196.5 |
| X-CLIP | Solo LoRA | 0.919 | **0.112** | **42.2** | **35.9** |
| Video Swin | En ejecución | — | — | — | 281.9 |

Umbral de tiempo real: ≤ 533 ms/clip (30 FPS, stride 16 frames).  
Los tres codificadores completados operan dentro del umbral.

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
