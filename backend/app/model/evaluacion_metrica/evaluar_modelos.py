#!/usr/bin/env python3
"""
Evaluación de métricas para modelos LoRA FLUX
Métricas: FID, KID, SSIM, PSNR, LPIPS, CLIP Score, DINO Score
Las métricas par a par (PSNR, SSIM, LPIPS, DINO) usan imágenes emparejadas
por nombre: 0001_chancay_0058_s0_7.png → chancay_0058.png
FID y KID usan las distribuciones completas de imágenes reales y generadas.
"""

import re
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
import json
from datetime import datetime


# ── helpers ──────────────────────────────────────────────────────────────────

def load_paired_images(real_dir, gen_dir, size=(256, 256)):
    """
    Carga imágenes emparejadas: para cada imagen generada busca
    su imagen real original usando el nombre embebido en el filename.
    Ejemplo: 0001_chancay_0058_s0_7.png → chancay_0058.png
    """
    real_dir = Path(real_dir)
    gen_dir  = Path(gen_dir)

    PATTERN = re.compile(r"^\d{4}_(.+)_s0_7\.png$")

    real_images, gen_images = [], []
    saltadas = 0

    gen_paths = sorted(gen_dir.glob("*.png"))
    for gen_path in gen_paths:
        match = PATTERN.match(gen_path.name)
        if not match:
            continue

        real_stem = match.group(1)           # ej: chancay_0058
        real_path = real_dir / f"{real_stem}.png"

        if not real_path.exists():
            saltadas += 1
            continue

        gen_img  = Image.open(gen_path).convert("RGB").resize(size)
        real_img = Image.open(real_path).convert("RGB").resize(size)

        gen_images.append(np.array(gen_img))
        real_images.append(np.array(real_img))

    print(f"  Pares encontrados : {len(real_images)}")
    if saltadas:
        print(f"  Saltadas (sin par): {saltadas}")
    return real_images, gen_images


def load_images_from_dir(directory, size=(256, 256)):
    """Carga todas las imágenes PNG de un directorio (para FID/KID)."""
    directory = Path(directory)
    images = []
    paths = sorted(directory.glob("*.png")) + sorted(directory.glob("*.jpg"))
    for p in paths:
        img = Image.open(p).convert("RGB").resize(size)
        images.append(np.array(img))
    print(f"  Cargadas {len(images)} imágenes de {directory.name}")
    return images


def images_to_tensor(images):
    """Convierte lista de numpy arrays a tensor [N, C, H, W] normalizado [0,1]."""
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
    """FID y KID usando torchmetrics (distribuciones completas, no par a par)."""
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


def calc_lpips(real_images, gen_images, device):
    """LPIPS (similitud perceptual, par a par)."""
    import lpips

    loss_fn = lpips.LPIPS(net="alex").to(device)
    real_tensor = images_to_tensor(real_images)
    gen_tensor  = images_to_tensor(gen_images)

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

    batch_size = 8
    for i in range(0, len(pil_images), batch_size):
        batch_imgs    = pil_images[i:i+batch_size]
        batch_prompts = (prompts[i:i+batch_size]
                         if len(prompts) > 1
                         else prompts * len(batch_imgs))
        inputs = processor(
            text=batch_prompts, images=batch_imgs,
            return_tensors="pt", padding=True, truncation=True
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits  = outputs.logits_per_image.diag()
            scores.extend(logits.cpu().tolist())

    return {
        "CLIP_score_mean": float(np.mean(scores)),
        "CLIP_score_std":  float(np.std(scores)),
    }


def calc_dino_score(real_images, gen_images, device):
    """
    DINO Score: similitud de características visuales par a par.
    Usado en DreamBooth para medir fidelidad de sujeto (Ruiz et al., 2023).
    Calcula similitud coseno entre embeddings ViT-S/16 de DINO.
    Rango: [-1, 1], donde 1 = idéntico. Valores típicos: 0.6–0.85.
    """
    from transformers import ViTModel, ViTFeatureExtractor

    print("  Cargando modelo DINO (facebook/dino-vits16)...")
    feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/dino-vits16")
    model = ViTModel.from_pretrained("facebook/dino-vits16").to(device)
    model.eval()

    def get_embeddings(images_np):
        pil_imgs = [Image.fromarray(img) for img in images_np]
        embeddings = []
        batch_size = 8
        for i in range(0, len(pil_imgs), batch_size):
            batch  = pil_imgs[i:i+batch_size]
            inputs = feature_extractor(images=batch, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :]  # token [CLS]
                embeddings.append(cls_emb.cpu())
        return torch.cat(embeddings, dim=0)

    n = min(len(real_images), len(gen_images))
    real_embs = get_embeddings(real_images[:n])
    gen_embs  = get_embeddings(gen_images[:n])

    cos_sim = torch.nn.functional.cosine_similarity(real_embs, gen_embs, dim=1)
    vals = cos_sim.tolist()

    return {
        "DINO_mean": float(np.mean(vals)),
        "DINO_std":  float(np.std(vals)),
    }


# ── evaluación principal ──────────────────────────────────────────────────────

def evaluate_model(version, real_dir, gen_dir, prompts, device, size=(256, 256)):
    print(f"\n{'='*60}")
    print(f"  Evaluando modelo: {version}")
    print(f"{'='*60}")

    # Imágenes emparejadas para métricas par a par
    print("\n[1/6] Cargando pares de imágenes...")
    real_paired, gen_paired = load_paired_images(real_dir, gen_dir, size)

    if len(real_paired) == 0 or len(gen_paired) == 0:
        print("  Sin pares encontrados, saltando.")
        return {}

    # Distribuciones completas para FID/KID
    print("\n  Cargando distribuciones completas para FID/KID...")
    real_all = load_images_from_dir(real_dir, size)
    gen_all  = load_images_from_dir(gen_dir,  size)

    real_tensor_all = images_to_tensor(real_all)
    gen_tensor_all  = images_to_tensor(gen_all)

    results = {
        "version":    version,
        "timestamp":  datetime.now().isoformat(),
        "n_pares":    len(real_paired),
        "n_real_all": len(real_all),
        "n_gen_all":  len(gen_all),
    }

    print("\n[2/6] Calculando PSNR y SSIM (par a par)...")
    results.update(calc_psnr_ssim(real_paired, gen_paired))

    print("\n[3/6] Calculando FID y KID (distribuciones completas)...")
    results.update(calc_fid_kid(real_tensor_all, gen_tensor_all, device))

    print("\n[4/6] Calculando LPIPS (par a par)...")
    results.update(calc_lpips(real_paired, gen_paired, device))

    print("\n[5/6] Calculando CLIP Score...")
    results.update(calc_clip_score(gen_paired, prompts, device))

    print("\n[6/6] Calculando DINO Score (par a par)...")
    results.update(calc_dino_score(real_paired, gen_paired, device))

    return results


def print_results(results):
    print(f"\n{'='*60}")
    print(f"  RESULTADOS — {results.get('version', '?')}")
    print(f"{'='*60}")
    print(f"  Pares emparejados  : {results.get('n_pares')}")
    print(f"  Reales (FID/KID)   : {results.get('n_real_all')}")
    print(f"  Generadas (FID/KID): {results.get('n_gen_all')}")
    print(f"\n  PSNR  : {results.get('PSNR_mean', 0):.4f} ± {results.get('PSNR_std', 0):.4f} dB  (↑ mejor)")
    print(f"  SSIM  : {results.get('SSIM_mean', 0):.4f} ± {results.get('SSIM_std', 0):.4f}       (↑ mejor)")
    print(f"  LPIPS : {results.get('LPIPS_mean', 0):.4f} ± {results.get('LPIPS_std', 0):.4f}       (↓ mejor)")
    print(f"  FID   : {results.get('FID', 0):.4f}                              (↓ mejor)")
    print(f"  KID   : {results.get('KID_mean', 0):.6f} ± {results.get('KID_std', 0):.6f}  (↓ mejor)")
    print(f"  CLIP  : {results.get('CLIP_score_mean', 0):.4f} ± {results.get('CLIP_score_std', 0):.4f}       (↑ mejor)")
    print(f"  DINO  : {results.get('DINO_mean', 0):.4f} ± {results.get('DINO_std', 0):.4f}       (↑ mejor)")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluación de métricas para modelos LoRA FLUX")
    parser.add_argument("--versions",  nargs="+", default=["v2", "v3", "v4", "v5"],
                        help="Versiones a evaluar, ej: v2 v3 v4 v5")
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
            print(f"\n  No existe {gen_dir}, saltando {version}.")
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
    print(f"\n  Resultados guardados en: {args.output}")

    # Comparación entre modelos
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  COMPARACIÓN ENTRE MODELOS")
        print(f"{'='*60}")
        print(f"  {'Métrica':<16}", end="")
        for r in all_results:
            print(f"  {r['version']:>10}", end="")
        print()
        metrics = [
            ("FID",             "↓ mejor"),
            ("KID_mean",        "↓ mejor"),
            ("SSIM_mean",       "↑ mejor"),
            ("PSNR_mean",       "↑ mejor"),
            ("LPIPS_mean",      "↓ mejor"),
            ("CLIP_score_mean", "↑ mejor"),
            ("DINO_mean",       "↑ mejor"),
        ]
        for metric, direction in metrics:
            print(f"  {metric:<16}", end="")
            for r in all_results:
                print(f"  {r.get(metric, 0):>10.4f}", end="")
            print(f"  {direction}")


if __name__ == "__main__":
    main()