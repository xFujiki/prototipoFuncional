Generación de Patrones Textiles Precolombinos
mediante Aprendizaje Profundo e Inteligencia Artificial Generativa

Proyecto de tesis — Pontificia Universidad Católica del Perú (PUCP)
Autor: Adrian Fujiki


Descripción
Este proyecto desarrolla un prototipo funcional capaz de generar imágenes de patrones textiles precolombinos andinos utilizando modelos de difusión de última generación. A través del fine-tuning con técnicas LoRA (Low-Rank Adaptation) sobre el modelo FLUX.1-dev, el sistema aprende el estilo visual del patrimonio textil cultural andino y es capaz de generar nuevas piezas con coherencia estética y semántica.
El prototipo incluye una interfaz web completa, pipeline de preprocesamiento de dataset, entrenamiento experimental con múltiples configuraciones y evaluación cuantitativa de calidad generativa.

Arquitectura del sistema
prototipoFuncional/
├── backend/                  # API REST con FastAPI
│   └── app/
│       ├── model/
│       │   ├── LoRA_FLUX.1/  # Scripts de inferencia y entrenamiento LoRA
│       │   ├── gan/          # Implementación StyleGAN2
│       │   └── evaluacion_metrica/  # Scripts de evaluación (FID, KID, CLIP, LPIPS...)
│       └── preprocessing_dataset/  # Aumentación y generación de captions
├── frontend/                 # Interfaz de usuario (React)
├── diffusers/                # Framework de difusión (HuggingFace)
└── training_flux_v*.log      # Logs de entrenamiento por versión

Experimentos de entrenamiento
Se entrenaron 5 versiones del modelo LoRA con distintas configuraciones de hiperparámetros sobre un dataset de textiles precolombinos andinos (784 imágenes con aumentación):
VersiónLoRA RankStepsLRSchedulerNovedad principalv2163 0001e-4constantLínea basev3324 0005e-5cosineMayor rank + decayv4163 0005e-5cosine restartsBatch size ×4v5645 0003e-5cosineText encoder entrenado
Resultados de evaluación
Métricav2v3v4v5MejorFID ↓141.06134.43136.31137.17v3KID ↓0.011300.010950.012080.01298v3LPIPS ↓0.74230.73920.73880.7393v4CLIP Score ↑28.9028.8428.8728.89v2 ≈ v5

Stack tecnológico
CapaTecnologíaModelo generativoFLUX.1-dev + LoRA (DreamBooth)Framework de difusiónHuggingFace DiffusersArquitectura alternativaStyleGAN2-ADABackendFastAPI (Python)FrontendReactEvaluacióntorchmetrics, lpips, CLIP, scikit-imageEntrenamientoPyTorch, bf16, CUDASeguimientoWeights & Biases (wandb)

Instalación y uso
Requisitos previos

Python 3.10+
CUDA 12.x
GPU con al menos 16 GB VRAM

Backend
bashgit clone https://github.com/xFujiki/prototipoFuncional.git
cd prototipoFuncional/backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
Frontend
bashcd frontend
npm install
npm run dev
Evaluación de métricas
bashcd backend/app/model/evaluacion_metrica
python evaluar_modelos.py \
  --versions v3 v4 \
  --real_dir /ruta/al/dataset \
  --gen_base /ruta/a/LoRA_FLUX.1 \
  --output ./metricas.json

Prompt de entrenamiento
pre-Columbian Andean textile, museum quality, cultural heritage artifact

Archivos importantes
ArchivoDescripciónbackend/app/model/evaluacion_metrica/evaluar_modelos.pyScript de evaluación de métricasbackend/app/model/LoRA_FLUX.1/inference_lora.pyInferencia con modelo LoRAbackend/app/preprocessing_dataset/augmentation.pyAumentación del datasetbackend/app/preprocessing_dataset/generate_captions_flux.pyGeneración de captionstraining_flux_v*.logLogs completos de cada entrenamiento

Contexto académico
Título de tesis:
"Generación de patrones textiles utilizando un modelo de aprendizaje profundo y técnicas de inteligencia artificial"
Universidad: Pontificia Universidad Católica del Perú (PUCP)
Facultad: Ciencias e Ingeniería
