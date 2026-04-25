import torch
from diffusers import FluxImg2ImgPipeline
from PIL import Image
from pathlib import Path

# ---- CONFIGURACIÓN ----
MODEL_ID    = "black-forest-labs/FLUX.1-dev"
LORA_PATH   = "/home/afujiki/prototipoFuncional/backend/app/model/LoRA_FLUX.1/lora_output"
IMAGE_PATH  = "/home/afujiki/prototipoFuncional/backend/app/model/LoRA_FLUX.1/train_images/inka_0012.png" #cambiar aqui la imagen input
OUTPUT_DIR  = "/home/afujiki/prototipoFuncional/backend/app/model/LoRA_FLUX.1/resultados_lora"

PROMPT   = (
    "pre-Columbian Andean textile, Inka culture, zoo pattern, add some animals" #cambiar este prompt, debe coincidir con el textil de arriba
    "polychrome wool, museum quality, high detail, cultural heritage artifact, zoo pattern"
)
STRENGTH = 0.5

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ---- CARGAR MODELO + LORA ----
print("Cargando FLUX.1-dev...")
pipe = FluxImg2ImgPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)

print("Cargando LoRA entrenada...")
pipe.load_lora_weights(LORA_PATH)
pipe.enable_model_cpu_offload()

print("Modelos listos.\n")

# ---- INFERENCIA ----
input_image = Image.open(IMAGE_PATH).convert("RGB").resize((512, 512))
input_image.save(f"{OUTPUT_DIR}/original.png")

for strength in [0.3, 0.5, 0.7]:
    print(f"Generando strength={strength}...")
    result = pipe(
        prompt=PROMPT,
        image=input_image,
        strength=strength,
        num_inference_steps=28,
        guidance_scale=3.5,
        generator=torch.Generator("cpu").manual_seed(42),
    ).images[0]

    out_path = f"{OUTPUT_DIR}/lora_strength_{str(strength).replace('.','_')}.png"
    result.save(out_path)
    print(f"  [SAVED] {out_path}")

print(f"\nListo. Resultados en: {OUTPUT_DIR}")