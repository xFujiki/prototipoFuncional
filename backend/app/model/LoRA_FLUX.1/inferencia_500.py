#!/usr/bin/env python3
"""
Script de inferencia por lotes — 500 imágenes por modelo LoRA FLUX
Genera imágenes con strength 0.75 (principal) y 0.5 (comparación)
Uso:
    python3 inferencia_500.py --version v3 --gpu 0
    python3 inferencia_500.py --version v3 --strength 0.5 --gpu 1
"""

import torch
import argparse
import random
from diffusers import FluxImg2ImgPipeline
from PIL import Image
from pathlib import Path

# ── Configuración base ────────────────────────────────────────────────────────

DATASET_DIR = Path("/home/afujiki/datasetPrecolombino/imagenes_256x256")
MODEL_ID    = "black-forest-labs/FLUX.1-dev"
BASE_OUTPUT = Path("/home/afujiki/prototipoFuncional/backend/app/model/LoRA_FLUX.1")

LORA_PATHS = {
    "v2": BASE_OUTPUT / "lora_output_flux",
    "v3": BASE_OUTPUT / "lora_output_flux_v3",
    "v4": BASE_OUTPUT / "lora_output_flux_v4",
    "v5": BASE_OUTPUT / "lora_output_flux_v5",
}

# Prompts por cultura
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

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_culture(filename: str) -> str:
    """Extrae la cultura del nombre de archivo (cultura_xxxx.png)."""
    return "_".join(filename.split("_")[:-1]).lower()


def build_image_pool(dataset_dir: Path, target: int, seed: int = 42) -> list:
    """
    Construye un pool de exactamente `target` imágenes del dataset.
    Si hay menos imágenes que target, repite aleatoriamente hasta completar.
    """
    all_imgs = sorted(dataset_dir.glob("*.png"))
    if len(all_imgs) == 0:
        raise FileNotFoundError(f"No se encontraron imágenes en {dataset_dir}")

    rng = random.Random(seed)

    if len(all_imgs) >= target:
        return rng.sample(all_imgs, target)

    # Repetir imágenes hasta completar el target
    pool = list(all_imgs)
    while len(pool) < target:
        pool += rng.sample(all_imgs, min(len(all_imgs), target - len(pool)))
    return pool[:target]


# ── Inferencia ────────────────────────────────────────────────────────────────

def run_inference(version: str, strength: float, gpu: int, n_images: int = 500):

    if version not in LORA_PATHS:
        raise ValueError(f"Versión '{version}' no reconocida. Opciones: {list(LORA_PATHS.keys())}")

    lora_path  = LORA_PATHS[version]
    output_dir = BASE_OUTPUT / f"resultados_lora_{version}"
    output_dir.mkdir(parents=True, exist_ok=True)

    strength_tag = str(strength).replace(".", "_")
    device = f"cuda:{gpu}"

    print(f"\n{'='*60}")
    print(f"  Modelo     : {version}")
    print(f"  LoRA path  : {lora_path}")
    print(f"  Output dir : {output_dir}")
    print(f"  Strength   : {strength}")
    print(f"  N imágenes : {n_images}")
    print(f"  Dispositivo: {device}")
    print(f"{'='*60}\n")

    # Cargar pipeline
    print("  Cargando modelo base FLUX.1-dev...")
    pipe = FluxImg2ImgPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(str(lora_path))
    pipe = pipe.to("cuda")
    print(f"  LoRA {version} cargada ✓\n")

    # Construir pool de imágenes
    image_pool = build_image_pool(DATASET_DIR, n_images, seed=42)
    print(f"  Pool de {len(image_pool)} imágenes construido ✓\n")

    # Generar
    generadas = 0
    saltadas  = 0

    for idx, img_path in enumerate(image_pool, 1):
        culture = get_culture(img_path.stem)
        prompt  = PROMPTS.get(culture, PROMPTS["inka"])  # fallback a inka

        # Nombre único: índice + nombre original + strength
        out_name = f"{idx:04d}_{img_path.stem}_s{strength_tag}.png"
        out_path = output_dir / out_name

        if out_path.exists():
            print(f"  [{idx:>3}/{n_images}] [SKIP] {out_name}")
            saltadas += 1
            continue

        # Procesar imagen
        input_image = Image.open(img_path).convert("RGB").resize((512, 512))

        result = pipe(
            prompt             = prompt,
            image              = input_image,
            strength           = strength,
            num_inference_steps= 28,
            guidance_scale     = 3.5,
            generator          = torch.Generator("cpu").manual_seed(42 + idx),
        ).images[0]

        result.save(out_path)
        generadas += 1
        print(f"  [{idx:>3}/{n_images}] [OK] {out_name}  ({culture})")

    # Limpiar memoria
    del pipe
    torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"  Modelo {version} completado")
    print(f"  Generadas : {generadas}")
    print(f"  Saltadas  : {saltadas} (ya existían)")
    print(f"  Output    : {output_dir}")
    print(f"{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inferencia por lotes — 500 imágenes por modelo LoRA FLUX")
    parser.add_argument("--version",  required=True, choices=["v2", "v3", "v4", "v5"],
                        help="Versión del modelo a usar")
    parser.add_argument("--strength", type=float, default=0.75,
                        help="Strength de img2img (default: 0.75 recomendado para evaluación)")
    parser.add_argument("--n",        type=int,   default=500,
                        help="Número de imágenes a generar (default: 500)")
    parser.add_argument("--gpu",      type=int,   default=0,
                        help="GPU a usar (default: 0)")
    args = parser.parse_args()

    run_inference(
        version  = args.version,
        strength = args.strength,
        gpu      = args.gpu,
        n_images = args.n,
    )


if __name__ == "__main__":
    main()