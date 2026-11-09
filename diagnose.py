"""Diagnóstico de la brecha VAL (0.50) vs score del test (0.19). Pensado para correr en Kaggle sobre un checkpoint ya entrenado.

  python diagnose.py --cache /kaggle/temp/train.npy --test <TEST> --ckpt runs/v1/last.pt

Prueba 1 (máscaras): accuracy en val con máscaras uniformes vs máscaras reales de target.csv.
Prueba 2 (fuga):     similitud de cada imagen de val / test con su vecino más parecido en train, y accuracy por nivel de similitud.
Usa el MISMO split aleatorio (seed 1234) con el que se entrenó el checkpoint viejo.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from common import DinoVQ, UNet, mask_from_target, random_masks, up_mask

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True, help="train.npy (uint8 N,256,256,3)")
ap.add_argument("--test", required=True)
ap.add_argument("--target", default="target.csv")
ap.add_argument("--codebook", default="codebook.npy")
ap.add_argument("--ckpt", default="runs/v1/last.pt")
ap.add_argument("--val-n", type=int, default=500)
a = ap.parse_args()
dev = "cuda"

data = np.load(a.cache, mmap_mode="r")
N = len(data)
perm = np.random.default_rng(1234).permutation(N)
val_idx, tr_idx = np.sort(perm[:a.val_n]), perm[a.val_n:]

dino = DinoVQ(a.codebook, dev)
net = UNet().to(dev).eval()
net.load_state_dict(torch.load(a.ckpt, map_location=dev)["ema"])

df = pd.read_csv(a.target)
bank = np.stack([mask_from_target(int(x) for x in str(t).split()) for t in df.target])          # (M,8,8)
files = {p.stem: p for p in Path(a.test).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}


def to_t(u8):                       # (B,H,W,3) uint8 -> (B,3,H,W) float en [0,1]
    return torch.from_numpy(np.ascontiguousarray(u8)).to(dev).permute(0, 3, 1, 2).float() / 255


@torch.no_grad()
def gvec(img, m8=None):
    """Vector global normalizado (media de los 64 embeddings de parche); con máscara, los parches ocultos se ponen en negro."""
    if m8 is not None:
        img = img * (1 - up_mask(m8))
    with torch.autocast("cuda", torch.float16):
        e = dino.embed(img)
    return F.normalize(e.float().mean(1), dim=-1)


def shift_img(img, c, g):
    """Más contraste (c) y brillo (g), como el test frente a train."""
    mu = img.mean((1, 2, 3), keepdim=True)
    return ((((img - mu) * c + mu) * g).clamp(0, 1) * 255).round() / 255


@torch.no_grad()
def eval_val(masks8, shift=None):
    """masks8: (n_val,8,8) bool; shift=(c,g) opcional. Devuelve accuracy global y por imagen."""
    hits, tots = [], []
    for i in range(0, len(val_idx), 25):
        img = to_t(data[val_idx[i:i + 25]])
        if shift:
            img = shift_img(img, *shift)
        m8 = torch.from_numpy(masks8[i:i + 25]).to(dev); m = up_mask(m8)
        with torch.autocast("cuda", torch.float16):
            gt = dino.codes(img)
            pred = net(img, m)
            pred = (pred.float().clamp(0, 1) * 255).round() / 255
            pc = dino.codes(pred)
        h = ((pc == gt) & m8.view(-1, 64)).sum(1); t = m8.view(-1, 64).sum(1)
        hits.append(h.cpu().numpy()); tots.append(t.cpu().numpy())
    h, t = np.concatenate(hits), np.concatenate(tots)
    return h.sum() / t.sum(), h / t


# ---------------------------------------------------------------- Prueba 1: tipo de máscara
g = torch.Generator(device=dev); g.manual_seed(999)
m_uni = random_masks(len(val_idx), dev, g).cpu().numpy()
rs = np.random.default_rng(0)
m_real = bank[rs.integers(len(bank), size=len(val_idx))]
acc_uni, _ = eval_val(m_uni)
acc_real, per_img = eval_val(m_real)
print("\n=== PRUEBA 1: máscaras ===")
print(f"VAL con máscaras uniformes (como en entrenamiento): {acc_uni:.4f}")
print(f"VAL con máscaras reales de target.csv             : {acc_real:.4f}")

# ---------------------------------------------------------------- Prueba 3: cambio fotométrico (el test es más brillante y con más contraste)
print("\n=== PRUEBA 3: cambio de brillo/contraste (máscaras reales) ===")
for c, g_ in ((1.0, 1.0), (1.4, 1.1), (1.8, 1.25), (2.2, 1.4)):
    acc, _ = eval_val(m_real, (c, g_))
    print(f"contraste x{c} brillo x{g_}: VAL acc = {acc:.4f}")

# ---------------------------------------------------------------- Prueba 2: fuga train/val
print("\n=== PRUEBA 2: similitud con el vecino más cercano en train (embedding global) ===")
tr_g = []
for i in range(0, len(tr_idx), 64):
    tr_g.append(gvec(to_t(data[np.sort(tr_idx)[i:i + 64]])))
tr_g = torch.cat(tr_g)                                                              # (Ntr, D)

va_g = torch.cat([gvec(to_t(data[val_idx[i:i + 25]]), torch.from_numpy(m_real[i:i + 25]).to(dev)) for i in range(0, len(val_idx), 25)])
te_g = []
for i in range(0, len(df), 25):
    ims, ms = [], []
    for r in df.iloc[i:i + 25].itertuples(index=False):
        ims.append(np.asarray(Image.open(files[str(r.ID)]).convert("RGB")))
        ms.append(mask_from_target(int(x) for x in str(r.target).split()))
    te_g.append(gvec(to_t(np.stack(ims)), torch.from_numpy(np.stack(ms)).to(dev)))
te_g = torch.cat(te_g)

va_ms = (va_g @ tr_g.T).max(1).values.cpu().numpy()
te_ms = (te_g @ tr_g.T).max(1).values.cpu().numpy()
q = lambda x: " ".join(f"{v:.3f}" for v in (x.mean(), *np.percentile(x, [10, 50, 90])))
print("sim. máx. a train        media   p10   p50   p90")
print(f"  val  (split aleatorio): {q(va_ms)}")
print(f"  test                  : {q(te_ms)}")

print("\naccuracy de val (máscaras reales) según qué tan parecida es la imagen a train:")
order = np.argsort(va_ms)
for name, sel in (("25% menos parecidas", order[:len(order) // 4]), ("25% más parecidas", order[-len(order) // 4:])):
    print(f"  {name}: sim={va_ms[sel].mean():.3f}  acc={per_img[sel].mean():.4f}")
lo = va_ms <= np.percentile(te_ms, 90)
print(f"  val con sim <= p90 del test: n={lo.sum()}  acc={per_img[lo].mean():.4f}")

print("\n=== LECTURA ===")
print("* Si Prueba 1 (reales) << uniformes -> las máscaras agrupadas del test explican la caída.")
print("* Si val tiene sim. mucho mayor que test, y acc(más parecidas) >> acc(menos parecidas) -> hay FUGA: la VAL está inflada.")
print("* Si Prueba 3 baja mucho al subir contraste/brillo (hacia ~0.2) -> el problema es el CAMBIO DE DOMINIO fotométrico: entrenar con --photo-aug.")
print("* Si ninguna de las dos explica la brecha -> diferencia de dominio entre train y test (o algo del pipeline de test).")
