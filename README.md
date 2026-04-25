<div align="center">

# Pre-Columbian Textile Generator

### Generación de Patrones Textiles Andinos mediante Aprendizaje Profundo e Inteligencia Artificial

**Tesis de pregrado — Pontificia Universidad Católica del Perú (PUCP)**

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-Vite-61DAFB?style=flat-square&logo=react&logoColor=black)
![FLUX.1](https://img.shields.io/badge/FLUX.1--dev-LoRA-8B5CF6?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)

</div>

---

## Descripción

Este repositorio contiene el prototipo funcional desarrollado como parte de la tesis *"Generación de patrones textiles utilizando un modelo de aprendizaje profundo y técnicas de inteligencia artificial"*. El sistema es una aplicación full-stack compuesta por un frontend en React y un backend en FastAPI, que aprovecha el modelo de difusión **FLUX.1-dev** fine-tuneado con **DreamBooth LoRA** para realizar generación imagen a imagen de patrones textiles precolombinos andinos (Inka, Wari, entre otros).

---

## Arquitectura del sistema

```
prototipoFuncional/
├── backend/                        # API REST (FastAPI)
│   └── app/
│       ├── model/
│       │   ├── LoRA_FLUX.1/        # Scripts de inferencia y evaluación
│       │   │   ├── inference_lora.py
│       │   │   ├── flux_img2img_test.py
│       │   │   └── evaluacion_metrica/
│       │   │       ├── evaluar_modelos.py
│       │   │       ├── metricas_v2_v3.json
│       │   │       ├── metricas_v4.json
│       │   │       └── metricas_v5.json
│       │   ├── gan/                # Arquitectura GAN (experimentos)
│       │   └── dataset.py
│       ├── preprocessing_dataset/
│       │   ├── augmentation.py
│       │   └── generate_captions_flux.py
│       └── main.py
├── frontend/                       # Interfaz web (React + Vite + Material-UI)
├── training_flux_v2.log            # Log de entrenamiento v2
├── training_flux_v3.log            # Log de entrenamiento v3
├── training_flux_v4.log            # Log de entrenamiento v4
└── training_flux_v5.log            # Log de entrenamiento v5
```

---

## Features

- **Full-Stack Application** — Interfaz web completa con cliente React y servidor FastAPI.
- **Image-to-Image Generation** — Genera nuevos patrones textiles a partir de una imagen fuente cargada por el usuario.
- **Fine-Tuned Diffusion Model** — Utiliza FLUX.1-dev adaptado con LoRA para resultados especializados en textiles precolombinos.
- **RESTful API** — Backend FastAPI con endpoint `/generate` escalable y eficiente.
- **Evaluación cuantitativa** — Script robusto para medir el rendimiento de distintas versiones del modelo con métricas estándar de la industria.

---

## Detalles del modelo

### FLUX.1 con LoRA

La generación de imágenes está impulsada por el modelo `black-forest-labs/FLUX.1-dev`, fine-tuneado mediante **Low-Rank Adaptation (LoRA)** con la técnica **DreamBooth** sobre un dataset personalizado de textiles precolombinos. Se entrenaron 4 versiones con diferentes configuraciones de hiperparámetros:

| Versión | LoRA Rank | Steps | Learning Rate | LR Scheduler | Novedad principal |
|---|---|---|---|---|---|
| **v2** | 16 | 3 000 | 1e-4 | constant | Línea base |
| **v3** | 32 | 4 000 | 5e-5 | cosine | Mayor capacidad + decay |
| **v4** | 16 | 3 000 | 5e-5 | cosine with restarts | Batch size efectivo ×4 |
| **v5** | 64 | 5 000 | 3e-5 | cosine | Text encoder entrenado |

> Todos los modelos usan `mixed_precision=bf16`, `seed=42`, resolución `256×256 px` y el prompt de instancia: *"pre-Columbian Andean textile, museum quality, cultural heritage artifact"*.

**Justificación de los cambios:** En v2 se estableció la línea base con lr constante y rank 16, lo que produjo inestabilidad en el loss y el peor FID (141.06). En v3 se redujo el lr a 5e-5, se duplicó el rank y se introdujo cosine decay, logrando la mejor distribución general (FID 134.43, KID 0.01095). En v4 se aumentó el batch size efectivo a 16 con cosine with restarts, obteniendo el menor loss final (0.458) y mejor LPIPS. Finalmente, v5 incorporó el entrenamiento del text encoder con lr independiente de 1e-5 y rank 64, alcanzando el CLIP Score más alto.

---

## Resultados de evaluación

Las métricas se calculan comparando las imágenes generadas contra el dataset real usando `evaluar_modelos.py`:

| Métrica | v2 | v3 | v4 | v5 | Mejor |
|---|---|---|---|---|---|
| **FID** ↓ | 141.06 | 134.43 | 136.31 | 137.17 | v3 |
| **KID** ↓ | 0.01130 | 0.01095 | 0.01208 | 0.01298 | v3 |
| **LPIPS** ↓ | 0.7423 | 0.7392 | 0.7388 | 0.7393 | v4 |
| **SSIM** ↑ | 0.0813 | 0.0847 | 0.0818 | 0.0818 | v3 |
| **PSNR** ↑ | 10.497 | 10.617 | 10.516 | 10.515 | v3 |
| **CLIP Score** ↑ | 28.90 | 28.84 | 28.87 | 28.89 | v2 |

> Evaluación realizada con 62 imágenes generadas por modelo. Se planea re-evaluar con 500 imágenes para mayor confianza estadística en el FID.

**Ranking general:** v3 > v4 ≈ v5 > v2

---

## Setup y ejecución

### Requisitos
- Python 3.10, Node.js 18+
- CUDA 12.x con GPU de al menos 16 GB VRAM

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

cd app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

La API estará disponible en `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

La interfaz estará disponible en `http://localhost:5173`.

---

## Evaluación de métricas

```bash
cd backend/app/model/evaluacion_metrica

python3 evaluar_modelos.py \
  --versions v3 v4 \
  --real_dir /ruta/dataset/imagenes_256x256 \
  --gen_base /ruta/LoRA_FLUX.1 \
  --output ./metricas_comparacion.json \
  --prompt "pre-Columbian Andean textile, museum quality, cultural heritage artifact" \
  --size 256 \
  --gpu 0
```

**Dependencias adicionales para evaluación:**
```bash
pip install torchmetrics lpips transformers scikit-image
```

---

## Uso

1. Asegúrate de que el backend y el frontend estén corriendo.
2. Abre `http://localhost:5173` en tu navegador.
3. Ve a la página **Generación** desde la barra de navegación.
4. Haz click en **"Seleccionar archivo"** para cargar una imagen de patrón textil.
5. Haz click en **Generar** para enviar la imagen al backend.
6. El resultado generado aparecerá en la parte derecha de la pantalla.

---

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| Modelo generativo | FLUX.1-dev + DreamBooth LoRA |
| Framework DL | PyTorch 2.x + Diffusers (HuggingFace) |
| Backend API | FastAPI + Uvicorn |
| Frontend | React + Vite + Material-UI |
| Evaluación | torchmetrics, lpips, transformers, scikit-image |
| Seguimiento | WandB |
| Entorno | Python 3.10, CUDA 12, bf16 |

---

## Archivos no incluidos en el repositorio

Por tamaño o confidencialidad, los siguientes recursos **no están incluidos**:

- Pesos del modelo (`*.safetensors`, `*.bin`)
- Checkpoints de entrenamiento (`lora_output*/`)
- Imágenes generadas (`resultados_lora*/`)
- Dataset de imágenes (`imagenes_*/`, `*.zip`)
- Entornos virtuales (`venv_flux/`, `venv/`)
- Repositorios externos clonados (`diffusers/`, `stylegan2-ada-pytorch/`)

---

## Autor

**Adrian Fujiki**  
Pontificia Universidad Católica del Perú — Facultad de Ciencias e Ingeniería  
GitHub: [@xFujiki](https://github.com/xFujiki)

---
