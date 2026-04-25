#!/usr/bin/env python3
"""
Evaluación de métricas para modelos LoRA FLUX
Métricas: FID, KID, SSIM, PSNR, LPIPS, CLIP Score
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
import json
from datetime import datetime

# ── helpers ──────────────────────────────────────────────────────────────────

def load_images_from_dir(directory, size=(256, 256)):
    """Carga todas las imágenes PNG de un directorio."""
    directory = Path(directory)
    images = []
    paths = sorted(directory.glob("*.png")) + sorted(directory.glob("*.jpg"))
    for p in paths:
        img = Image.open(p).convert("RGB").resize(size)
        images.append(np.array(img))
    print(f"  Cargadas {len(images)} imágenes de {directory}")
    return images, [str(p) for p in paths]


def images_to_tensor(images):
    """Convierte lista de numpy arrays a tensor [N, C, H, W] normalizado."""
    arr = np.stack(images).astype(np.float32) / 255.0
    return torch.tensor(arr).permute(0, 3, 1, 2)


# ── métricas individuales ─────────────────────────────────────────────────────

def calc_psnr_ssim(real_images, gen_images):
    """PSNR y SSIM imagen a imagen (emparejadas por índice)."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    psnr_vals, ssim_vals = [], []
    n = min(len(real_images), len(gen_images))
    for i in range(n):
        r = real_images[i].astype(np.float64)
        g = gen_images[i].astype(np.float64)
        psnr_vals.append(peak_signal_noise_ratio(r, g, data_range=255))
        ssim_vals.append(structural_similarity(r, g, channel_axis=2, data_range=255))

    return {
        "PSNR_mean": float(np.mean(psnr_vals)),
        "PSNR_std":  float(np.std(psnr_vals)),
        "SSIM_mean": float(np.mean(ssim_vals)),
        "SSIM_std":  float(np.std(ssim_vals)),
    }


def calc_fid_kid(real_tensor, gen_tensor, device):
    """FID y KID usando torchmetrics."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    real_uint8 = (real_tensor * 255).byte().to(device)
    gen_uint8  = (gen_tensor  * 255).byte().to(device)

    # FID
    fid = FrechetInceptionDistance(normalize=False).to(device)
    fid.update(real_uint8, real=True)
    fid.update(gen_uint8,  real=False)
    fid_score = float(fid.compute())

    # KID
    kid = KernelInceptionDistance(subset_size=min(50, len(real_uint8)), normalize=False).to(device)
    kid.update(real_uint8, real=True)
    kid.update(gen_uint8,  real=False)
    kid_mean, kid_std = kid.compute()

    return {
        "FID":      fid_score,
        "KID_mean": float(kid_mean),
        "KID_std":  float(kid_std),
    }


def calc_lpips(real_tensor, gen_tensor, device):
    """LPIPS (similitud perceptual)."""
    import lpips
    loss_fn = lpips.LPIPS(net="alex").to(device)

    # normalizar a [-1, 1]
    r = (real_tensor * 2 - 1).to(device)
    g = (gen_tensor  * 2 - 1).to(device)

    n = min(len(r), len(g))
    vals = []
    with torch.no_grad():
        for i in range(n):
            vals.append(float(loss_fn(r[i:i+1], g[i:i+1])))

    return {
        "LPIPS_mean": float(np.mean(vals)),
        "LPIPS_std":  float(np.std(vals)),
    }


def calc_clip_score(gen_images, prompts, device):
    """CLIP Score: coherencia entre prompt e imagen generada."""
    from transformers import CLIPProcessor, CLIPModel

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    scores = []
    pil_images = [Image.fromarray(img) for img in gen_images]

    # Procesar en batches de 8
    batch_size = 8
    for i in range(0, len(pil_images), batch_size):
        batch_imgs    = pil_images[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size] if len(prompts) > 1 else prompts * len(batch_imgs)
        inputs = processor(text=batch_prompts, images=batch_imgs,
                           return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits  = outputs.logits_per_image.diag()
            scores.extend(logits.cpu().tolist())

    return {
        "CLIP_score_mean": float(np.mean(scores)),
        "CLIP_score_std":  float(np.std(scores)),
    }


# ── evaluación principal ──────────────────────────────────────────────────────

def evaluate_model(version, real_dir, gen_dir, prompts, device, size=(256, 256)):
    print(f"\n{'='*60}")
    print(f"  Evaluando modelo: {version}")
    print(f"{'='*60}")

    print("\n[1/5] Cargando imágenes...")
    real_images, _ = load_images_from_dir(real_dir,  size)
    gen_images,  _ = load_images_from_dir(gen_dir,   size)

    if len(real_images) == 0 or len(gen_images) == 0:
        print("  ⚠️  Sin imágenes suficientes, saltando.")
        return {}

    real_tensor = images_to_tensor(real_images)
    gen_tensor  = images_to_tensor(gen_images)

    results = {"version": version, "timestamp": datetime.now().isoformat(),
               "n_real": len(real_images), "n_gen": len(gen_images)}

    print("\n[2/5] Calculando PSNR y SSIM...")
    results.update(calc_psnr_ssim(real_images, gen_images))

    print("\n[3/5] Calculando FID y KID...")
    results.update(calc_fid_kid(real_tensor, gen_tensor, device))

    print("\n[4/5] Calculando LPIPS...")
    results.update(calc_lpips(real_tensor, gen_tensor, device))

    print("\n[5/5] Calculando CLIP Score...")
    results.update(calc_clip_score(gen_images, prompts, device))

    return results


def print_results(results):
    print(f"\n{'='*60}")
    print(f"  RESULTADOS — {results.get('version', '?')}")
    print(f"{'='*60}")
    print(f"  Imágenes reales : {results.get('n_real')}")
    print(f"  Imágenes gen.   : {results.get('n_gen')}")
    print(f"\n  PSNR  : {results.get('PSNR_mean', 0):.4f} ± {results.get('PSNR_std', 0):.4f} dB")
    print(f"  SSIM  : {results.get('SSIM_mean', 0):.4f} ± {results.get('SSIM_std', 0):.4f}")
    print(f"  LPIPS : {results.get('LPIPS_mean', 0):.4f} ± {results.get('LPIPS_std', 0):.4f}  (↓ mejor)")
    print(f"  FID   : {results.get('FID', 0):.4f}                    (↓ mejor)")
    print(f"  KID   : {results.get('KID_mean', 0):.6f} ± {results.get('KID_std', 0):.6f}  (↓ mejor)")
    print(f"  CLIP  : {results.get('CLIP_score_mean', 0):.4f} ± {results.get('CLIP_score_std', 0):.4f}  (↑ mejor)")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluación de métricas para modelos LoRA FLUX")
    parser.add_argument("--versions",  nargs="+", default=["v2", "v3"],
                        help="Versiones a evaluar, ej: v2 v3")
    parser.add_argument("--real_dir",  default="/home/afujiki/datasetPrecolombino/imagenes_256x256",
                        help="Directorio con imágenes reales del dataset")
    parser.add_argument("--gen_base",  default="/home/afujiki/prototipoFuncional/backend/app/model/LoRA_FLUX.1",
                        help="Directorio base donde están las carpetas resultados_lora_vX")
    parser.add_argument("--output",    default="./metricas_resultados.json",
                        help="Archivo JSON donde guardar los resultados")
    parser.add_argument("--prompt",    default="pre-Columbian Andean textile, museum quality, cultural heritage artifact",
                        help="Prompt usado para CLIP Score")
    parser.add_argument("--size",      type=int, default=256,
                        help="Resolución a la que redimensionar (default: 256)")
    parser.add_argument("--gpu",       type=int, default=0,
                        help="GPU a usar (default: 0)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")

    all_results = []

    for version in args.versions:
        gen_dir = Path(args.gen_base) / f"resultados_lora_{version}"
        if not gen_dir.exists():
            print(f"\n⚠️  No existe {gen_dir}, saltando {version}.")
            continue

        results = evaluate_model(
            version  = version,
            real_dir = args.real_dir,
            gen_dir  = str(gen_dir),
            prompts  = [args.prompt],
            device   = device,
            size     = (args.size, args.size),
        )
        if results:
            print_results(results)
            all_results.append(results)

    # Guardar JSON
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Resultados guardados en: {args.output}")

    # Comparación rápida si hay más de un modelo
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  COMPARACIÓN ENTRE MODELOS")
        print(f"{'='*60}")
        print(f"  {'Métrica':<12}", end="")
        for r in all_results:
            print(f"  {r['version']:>10}", end="")
        print()
        for metric in ["FID", "KID_mean", "SSIM_mean", "PSNR_mean", "LPIPS_mean", "CLIP_score_mean"]:
            print(f"  {metric:<12}", end="")
            for r in all_results:
                print(f"  {r.get(metric, 0):>10.4f}", end="")
            print()


if __name__ == "__main__":
    main()
