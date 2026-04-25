import torch
from diffusers import FluxImg2ImgPipeline
from PIL import Image
from pathlib import Path

# ---- CONFIGURACIÓN ----
MODEL_ID   = "black-forest-labs/FLUX.1-dev"
IMAGE_PATH = "/home/afujiki/prototipoFuncional/backend/app/model/LoRA_FLUX.1/imagenes_256_flux/wari_0001.png"
OUTPUT_DIR = "/home/afujiki/prototipoFuncional/backend/app/model/LoRA_FLUX.1/test_img2img"

STRENGTHS = [0.3, 0.5, 0.7]

PROMPT = (
    "pre-Columbian Andean textile, geometric pattern, "
    "polychrome wool, museum quality, high detail"
)

# ---- CARGAR MODELO ----
print("Cargando FLUX.1-dev img2img...")
pipe = FluxImg2ImgPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
print("Modelo cargado.\n")

# ---- INFERENCIA ----
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

input_image = Image.open(IMAGE_PATH).convert("RGB").resize((512, 512))
input_image.save(f"{OUTPUT_DIR}/original.png")
print("Imagen original guardada.")

for strength in STRENGTHS:
    print(f"Generando con strength={strength}...")

    result = pipe(
        prompt=PROMPT,
        image=input_image,
        strength=strength,
        num_inference_steps=28,
        guidance_scale=3.5,
        generator=torch.Generator("cpu").manual_seed(42),
    ).images[0]

    out_path = f"{OUTPUT_DIR}/strength_{str(strength).replace('.', '_')}.png"
    result.save(out_path)
    print(f"  [SAVED] {out_path}")

print(f"\nListo. Revisa los resultados en: {OUTPUT_DIR}")