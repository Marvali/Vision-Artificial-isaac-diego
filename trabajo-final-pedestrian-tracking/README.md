# Visión Artificial — Trabajo final: Pedestrian Tracking

Repositorio de la asignatura **Visión Artificial** (prácticas 2–6 y trabajo final), realizado por **Isaac Marval y Diego Peña**.

Este README se centra en el **trabajo final**, situado en la carpeta [`trabajo-final-pedestrian-tracking`](.).

---

## 1. Objetivo

Detectar y seguir (*tracking*) peatones en vídeo con MATLAB, y **comparar distintos detectores** dentro de un mismo pipeline de seguimiento:

| Detector | Tipo | Función de MATLAB |
|---|---|---|
| Red de personas de MathWorks | Deep learning | `peopleDetector('medium-network')` |
| ACF entrenado en Caltech | Clásico (Aggregate Channel Features) | `peopleDetectorACF('caltech-50x21')` |
| ACF entrenado en INRIA | Clásico (Aggregate Channel Features) | `peopleDetectorACF('inria-100x41')` |
| YOLOv4 (CSP-Darknet53, COCO) | Deep learning, con soporte GPU | `yolov4ObjectDetector('csp-darknet53-coco')` |

La calidad de cada detector se mide contra un *ground truth* con las métricas **IoU (Jaccard)** y **Dice**.

## 2. Punto de partida

Se parte del ejemplo oficial de MathWorks *Tracking Pedestrians from a Moving Car* (`PedestrianTrackingFromMovingCameraExample.m`, vídeo `vippedtracking.mp4`). Sobre él se ha:

1. Sustituido/ampliado el detector (ACF Caltech, ACF INRIA y YOLOv4).
2. Cambiado el escenario a un vídeo de cámara fija, **PETS09-S2L1**.
3. Añadido la evaluación cuantitativa (IoU y Dice) frente al ground truth.
4. Ajustado los parámetros del tracker para cada detector.

## 3. Pipeline

```
Frame ─► Detección ─► Filtro por escala ─► NMS ─► Asignación ─► Kalman ─► Gestión de tracks ─► Visualización
                                                                                       │
                                                   Ground truth (gt.txt) ──► IoU / Dice ┘
```

1. **Detección**: el detector devuelve cajas (`bboxes`). En ACF el frame se reescala (`resizeRatio` 1.5 para Caltech y 2 para INRIA) para detectar peatones pequeños. En YOLOv4 se filtra solo la clase `person` (umbral 0.45).
2. **Filtro por escala**: `pedScaleTable.mat` (generada con `helperTableOfScales.m`) da la altura esperada de un peatón según la posición de sus pies. Se descartan las cajas cuya altura se desvía más de `scThresh` (30–40 %).
3. **Supresión de no máximos**: `selectStrongestBbox` con solape mínimo 0.6.
4. **Predicción**: un **filtro de Kalman** de velocidad constante (`configureKalmanFilter('ConstantVelocity', …)`) por track.
5. **Asignación detección–track**: coste `1 − IoU` entre la caja predicha y la detectada, con *gating* y resolución con `assignDetectionsToTracks` (algoritmo húngaro).
6. **Gestión de tracks**: actualizar los asignados, avanzar con la predicción los no asignados, borrar tracks perdidos (edad alta y visibilidad baja) y crear nuevos con las detecciones sin asignar.
7. **Visualización**: se dibujan los tracks, y en las versiones con métricas también las detecciones crudas (rojo) y el ground truth (verde).

## 4. Datos

| Archivo | Descripción |
|---|---|
| `PETS09-S2L1-raw.webm` | Vídeo PETS09-S2L1 (cámara fija, peatones en una escena exterior). |
| `vippedtracking.mp4` | Vídeo del ejemplo de MathWorks (cámara en un coche en movimiento). |
| `gt.txt` | *Ground truth* en formato MOT: `frame, id, x, y, w, h, …`. |
| `detection.txt` | Detecciones de referencia en formato MOT. |
| `pedScaleTable.mat` | Tabla de altura esperada de un peatón por fila de la imagen. |

**Sincronización del ground truth:** el vídeo y `gt.txt` no van al mismo ritmo, así que el frame de GT se calcula como `round(frame × GT_RATIO) + GT_OFFSET` (con `GT_RATIO = 2.0` y `GT_OFFSET = 0`). Si se cambia de vídeo o de fuente, estos dos valores hay que reajustarlos.

## 5. Contenido de la carpeta

| Archivo | Qué hace |
|---|---|
| `PedestrianTrackingFromMovingCameraExample.m` | Ejemplo base de MathWorks (cámara en movimiento, `vippedtracking.mp4`). |
| `PedestrianTrackingFromMovingCameraExample_CALTECH.mlx` | Versión con detector ACF Caltech. |
| `PedestrianTrackingFromMovingCameraYOLO4GPU.mlx` | Versión con **YOLOv4** y uso de GPU si está disponible (si no, CPU). |
| `PedestrianTrackingYOLOv4_GPU_DIAG.m` | Versión de diagnóstico de YOLOv4 en GPU. |
| `PedestrianTracking_Caltech_Metrics.mlx` | Tracking con ACF Caltech **y cálculo de IoU/Dice**. |
| `inra.mlx` | Tracking con ACF INRIA. |
| `inrametric.mlx` | Tracking con ACF INRIA **y cálculo de IoU/Dice**. |
| `helperTableOfScales.m` | Genera la tabla de escalas `pedScaleTable.mat`. |
| `prueba.m` | Script de pruebas. |

## 6. Parámetros principales

| Parámetro | Significado | Caltech | INRIA | YOLOv4 |
|---|---|---|---|---|
| `ROI` | Zona de procesado `[x y w h]` | `[]` (toda la imagen) | `[]` | `[40 95 400 140]` |
| `scThresh` | Tolerancia en el filtro de escala | 0.4 | 0.3 | 0.3 |
| `gatingThresh` | Umbral de coste para permitir una asignación | 0.7 | 0.7 | 0.9 |
| `costOfNonAssignment` | Coste de no asignar | 20 | 20 | 10 |
| `ageThresh` | Edad mínima para evaluar un track | 4 | 4 | 3 |
| `visThresh` | Visibilidad mínima durante la vida del track | 0.7 | 0.7 | 0.6 |

## 7. Métricas

Para cada detección se busca la caja de *ground truth* con la que más se solapa y se calcula:

- **IoU (Jaccard)**: `bboxOverlapRatio(det, gt)`.
- **Dice**: `dice(maskDet, maskGT)`, con máscaras rectangulares generadas desde las cajas (`rect2mask`).

Al terminar el vídeo se imprime por consola el **Mean IoU** y el **Mean Dice** (si una detección no se solapa con nada, aporta 0). Durante la ejecución se muestran las medias acumuladas sobre el vídeo.

## 8. Requisitos

- **MATLAB** con:
  - Computer Vision Toolbox
  - Image Processing Toolbox
  - Deep Learning Toolbox
  - *Computer Vision Toolbox Model for YOLO v4 Object Detection* (Add-On Explorer, solo para la versión YOLOv4)
  - Parallel Computing Toolbox y una GPU NVIDIA compatible con CUDA (opcional; sin ella YOLOv4 funciona en CPU, más lento)

## 9. Cómo ejecutarlo

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/Marvali/Vision-Artificial-isaac-diego.git
   ```
2. En MATLAB, abrir (o poner como carpeta actual) `trabajo-final-pedestrian-tracking`. Los scripts leen los vídeos, `gt.txt` y `pedScaleTable.mat` desde el directorio actual.
3. Ejecutar el script que se quiera:
   - Detector Caltech con métricas: `PedestrianTracking_Caltech_Metrics.mlx`
   - Detector INRIA con métricas: `inrametric.mlx`
   - Detector YOLOv4: `PedestrianTrackingFromMovingCameraYOLO4GPU.mlx`
   - Ejemplo base: `PedestrianTrackingFromMovingCameraExample`
4. Para cambiar de vídeo, editar `videoFile` (`PETS09-S2L1-raw.webm` o `vippedtracking.mp4`) y, si hace falta, `GT_RATIO` / `GT_OFFSET`.

## 10. Resultados

Los vídeos se reproducen en un `vision.VideoPlayer` con:

- **Verde**: ground truth.
- **Rojo**: detecciones crudas del detector.
- Cajas con ID y color propio por track: resultado del tracker.

> Las cifras finales de Mean IoU / Mean Dice de cada detector salen por consola al terminar la ejecución; añádelas aquí cuando tengas los números definitivos.

| Detector | Mean IoU | Mean Dice |
|---|---|---|
| ACF Caltech | _por completar_ | _por completar_ |
| ACF INRIA | _por completar_ | _por completar_ |
| YOLOv4 | _por completar_ | _por completar_ |

## 11. Créditos

- Autores: **Isaac Marval** y **Diego Peña**.
- Ejemplo base: *Tracking Pedestrians from a Moving Car*, © MathWorks.
- Vídeo y anotaciones: dataset **PETS 2009** (S2.L1).
- Modelos: Caltech y INRIA (ACF), YOLOv4 (COCO), distribuidos por MathWorks.
