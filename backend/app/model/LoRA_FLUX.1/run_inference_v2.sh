#!/bin/bash
GPU=${1:-1}

VERSION="v2"
STRENGTH=0.7

echo "============================================================"
echo "  Modelo   : $VERSION"
echo "  Strength : $STRENGTH"
echo "  GPU      : $GPU"
echo "  Seed     : ALEATORIO"
echo "  Imágenes : 100 (aleatorias)"
echo "============================================================"

cd ~/prototipoFuncional
source venv_flux/bin/activate

CUDA_VISIBLE_DEVICES=$GPU python3 - << PYEOF
import torch
import random
from diffusers import FluxImg2ImgPipeline
from PIL import Image
from pathlib import Path

VERSION      = "v2"
STRENGTH     = 0.7
N_IMAGES     = 100
SEED         = random.randint(0, 2**32 - 1)   # Seed global aleatorio
DATASET_DIR  = Path("/home/afujiki/datasetPrecolombino/imagenes_256x256")
BASE         = Path("/home/afujiki/prototipoFuncional/backend/app/model/LoRA_FLUX.1")
MODEL_ID     = "black-forest-labs/FLUX.1-dev"

LORA_PATHS = {
    "v2": BASE / "lora_output_flux",
    "v3": BASE / "lora_output_flux_v3",
    "v4": BASE / "lora_output_flux_v4",
    "v5": BASE / "lora_output_flux_v5",
}

PROMPTS = {
    "paracas":     "pre-Columbian Andean textile, Paracas culture, geometric pattern, polychrome wool, museum quality, cultural heritage artifact",
    "nazca":       "pre-Columbian Andean textile, Nazca culture, geometric pattern, polychrome wool, museum quality, cultural heritage artifact",
    "wari":        "pre-Columbian Andean textile, Wari culture, geometric pattern, polychrome wool, museum quality, cultural heritage artifact",
    "inka":        "pre-Columbian Andean textile, Inka culture, geometric pattern, polychrome wool, museum quality, cultural heritage artifact",
    "chimu":       "pre-Columbian Andean textile, Chimu culture, geometric pattern, polychrome wool, museum quality, cultural heritage artifact",
    "chancay":     "pre-Columbian Andean textile, Chancay culture, geometric pattern, polychrome wool, museum quality, cultural heritage artifact",
    "chavin":      "pre-Columbian Andean textile, Chavin culture, geometric pattern, polychrome wool, museum quality, cultural heritage artifact",
    "lambayeque":  "pre-Columbian Andean textile, Lambayeque culture, geometric pattern, polychrome wool, museum quality, cultural heritage artifact",
    "moche":       "pre-Columbian Andean textile, Moche culture, geometric pattern, polychrome wool, museum quality, cultural heritage artifact",
    "tiwanaku":    "pre-Columbian Andean textile, Tiwanaku culture, geometric pattern, polychrome wool, museum quality, cultural heritage artifact",
    "nazca_huari": "pre-Columbian Andean textile, Nazca-Huari culture, geometric pattern, polychrome wool, museum quality, cultural heritage artifact",
    "arica":       "pre-Columbian Andean textile, Arica culture, geometric pattern, polychrome wool, museum quality, cultural heritage artifact",
    "lima":        "pre-Columbian Andean textile, Lima culture, geometric pattern, polychrome wool, museum quality, cultural heritage artifact",
    "recuay":      "pre-Columbian Andean textile, Recuay culture, geometric pattern, polychrome wool, museum quality, cultural heritage artifact",
    "siguas":      "pre-Columbian Andean textile, Siguas culture, geometric pattern, polychrome wool, museum quality, cultural heritage artifact",
    "quechua":     "pre-Columbian Andean textile, Quechua culture, geometric pattern, polychrome wool, museum quality, cultural heritage artifact",
}

DEFAULT_PROMPT = "pre-Columbian Andean textile, museum quality, cultural heritage artifact"

def get_culture(stem):
    parts = stem.split("_")
    if len(parts) >= 3 and not parts[1].isdigit():
        return f"{parts[0]}_{parts[1]}"
    return parts[0]

def build_pool(dataset_dir, target, seed):
    all_imgs = sorted(dataset_dir.glob("*.png"))
    if not all_imgs:
        raise FileNotFoundError(f"No hay imagenes en {dataset_dir}")
    rng = random.Random(seed)
    # Siempre muestreo sin reemplazo si hay suficientes; si no, rellena
    if len(all_imgs) >= target:
        return rng.sample(all_imgs, target)
    pool = list(all_imgs)
    while len(pool) < target:
        pool += rng.sample(all_imgs, min(len(all_imgs), target - len(pool)))
    return pool[:target]

lora_path  = LORA_PATHS[VERSION]
output_dir = Path.cwd() / "inferencias_form_v2_paralelo"
output_dir.mkdir(parents=True, exist_ok=True)
strength_tag = str(STRENGTH).replace(".", "_")

print(f"\n{'='*60}")
print(f"  Seed global : {SEED}")
print(f"  Output      : {output_dir}")
print(f"  Cargando modelo {VERSION}...")
print(f"{'='*60}")

pipe = FluxImg2ImgPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(str(lora_path))
pipe.enable_model_cpu_offload()
print(f"  LoRA {VERSION} cargada\n")

image_pool = build_pool(DATASET_DIR, N_IMAGES, SEED)
print(f"  Pool de {len(image_pool)} imágenes aleatorias construido\n")

generadas = 0
saltadas  = 0

for idx, img_path in enumerate(image_pool, 1):
    culture  = get_culture(img_path.stem)
    prompt   = PROMPTS.get(culture, DEFAULT_PROMPT)
    out_name = f"{idx:04d}_{img_path.stem}_s{strength_tag}_seed{SEED}.png"
    out_path = output_dir / out_name

    if out_path.exists():
        print(f"  [{idx:>3}/{N_IMAGES}] [SKIP] {out_name}")
        saltadas += 1
        continue

    input_image = Image.open(img_path).convert("RGB").resize((512, 512))
    result = pipe(
        prompt=prompt,
        image=input_image,
        strength=STRENGTH,
        num_inference_steps=28,
        guidance_scale=3.5,
        generator=torch.Generator("cpu").manual_seed(SEED + idx),
    ).images[0]

    result.save(out_path)
    generadas += 1
    print(f"  [{idx:>3}/{N_IMAGES}] [OK] {out_name}  ({culture})")

del pipe
torch.cuda.empty_cache()

print(f"\n{'='*60}")
print(f"  {VERSION} completado — {generadas} generadas, {saltadas} saltadas")
print(f"  Seed usado  : {SEED}")
print(f"  Output      : {output_dir}")
print(f"{'='*60}\n")
PYEOF