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

| Versión | LoRA Rank | Steps | Learning Rate | LR Scheduler | Loss final | Novedad principal |
|---|---|---|---|---|---|---|
| **v2** | 16 | 3 000 | 1e-4 | constant | 0.527 | Línea base |
| **v3** | 32 | 4 000 | 5e-5 | cosine | 0.478 | Mayor capacidad + decay |
| **v4** | 16 | 3 000 | 5e-5 | cosine with restarts | 0.458 | Batch size efectivo ×4 |
| **v5** | 64 | 5 000 | 3e-5 | cosine | 0.468 | Text encoder entrenado |

> Todos los modelos usan `mixed_precision=bf16`, `seed=42`, resolución `256×256 px` y el prompt de instancia: *"pre-Columbian Andean textile, museum quality, cultural heritage artifact"*.

**Justificación de los cambios:** En v2 se estableció la línea base con lr constante y rank 16. En v3 se redujo el lr a 5e-5, se duplicó el rank y se introdujo cosine decay. En v4 se aumentó el batch size efectivo a 16 mediante acumulación de gradiente, logrando el menor loss final (0.458). Finalmente, v5 incorporó el entrenamiento del text encoder con lr independiente de 1e-5 y rank 64, buscando mayor alineación entre prompts textuales e imágenes generadas.

---

## Resultados de evaluación

### Evaluación cuantitativa

Las métricas se calculan comparando 500 inferencias por modelo contra el dataset real usando `evaluar_modelos.py`, con un parámetro `strength=0.70`:

| Métrica | v2 | v3 | v4 | v5 | Mejor |
|---|---|---|---|---|---|
| **FID** ↓ | 55.558 | 60.238 | 61.089 | 59.880 | **v2** |
| **KID** ↓ | 0.00482 | 0.00803 | 0.00839 | 0.00843 | **v2** |
| **LPIPS** ↓ | 0.3720 | 0.3832 | 0.3849 | 0.3833 | **v2** |
| **CLIP Score** ↑ | 28.169 | 28.094 | 28.101 | 28.137 | **v2** |
| **DINO** ↑ | 0.7542 | 0.7262 | 0.7262 | 0.7265 | **v2** |

**Ranking general:** v2 > v5 ≈ v3 > v4

> v2 obtiene los mejores resultados en todas las métricas, indicando mayor fidelidad visual y coherencia texto-imagen respecto al dataset de referencia. Este resultado debe interpretarse con cuidado: una fidelidad elevada puede reflejar menor diversificación generativa. Las versiones con menor puntaje podrían estar exhibiendo mayor creatividad, aunque a costa de alejarse del dominio visual de referencia.

### Evaluación cualitativa

Se realizó una revisión visual de un subconjunto de las 500 inferencias por versión, comparando textiles de distintas culturas (Inka, Chimú, Nazca, Paracas, Chancay). Los resultados son coherentes y reconocibles como patrones textiles precolombinos en las cuatro versiones, sin diferencias perceptuales significativas entre ellas.

### Validación experta

Un subconjunto de 30 imágenes generadas por el modelo **v2** fue evaluado por **Milagritos Jiménez**, magíster en Arqueología y especialista del Museo de Arqueología Josefina Ramos de Cox, mediante un formulario con escala Likert del 1 al 5.

| Resultado | Valor |
|---|---|
| Imágenes reconocidas como textil precolombino (puntaje ≥ 3) | **66.7%** |
| IOV objetivo | 70% |

> El 66.7% se considera parcialmente satisfactorio dado que la validación se realizó con un único evaluador y sobre un subconjunto limitado. Ampliar el número de especialistas y el conjunto de imágenes podría conducir a resultados más representativos.

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
  --versions v2 v3 v4 v5 \
  --real_dir /ruta/dataset/imagenes_256x256 \
  --gen_base /ruta/LoRA_FLUX.1 \
  --output ./metricas_resultados.json \
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
- Imágenes generadas (`resultados_lora*/`, `image_test_lora_v*/`, `resultados_test_v*/`)
- Inferencias de formulario (`inferencias_form_v*/`)
- Dataset de imágenes (`imagenes_*/`, `*.zip`)
- Entornos virtuales (`venv_flux/`, `venv/`)
- Repositorios externos clonados (`diffusers/`, `stylegan2-ada-pytorch/`)

---

## Autor

**Adrian Fujiki**  
Pontificia Universidad Católica del Perú — Facultad de Ciencias e Ingeniería  
GitHub: [@xFujiki](https://github.com/xFujiki)

---
