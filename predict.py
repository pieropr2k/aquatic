"""Reconstruye test: pega la predicción SOLO en los patches ocultos y guarda PNG (mismo nombre) en --out."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from common import UNet, mask_from_target, up_mask

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--test", default="test")
ap.add_argument("--target", default="target.csv")
ap.add_argument("--out", default="generate")
ap.add_argument("--tta", action="store_true")
a = ap.parse_args()

dev = "cuda"
Path(a.out).mkdir(parents=True, exist_ok=True)
net = UNet().to(dev).eval()
net.load_state_dict(torch.load(a.ckpt, map_location=dev)["ema"])
df = pd.read_csv(a.target)
files = {p.stem: p for p in Path(a.test).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}

for r in tqdm(df.itertuples(index=False), total=len(df)):
    orig = np.asarray(Image.open(files[str(r.ID)]).convert("RGB"))
    m8 = mask_from_target(int(x) for x in str(r.target).split())
    img = torch.from_numpy(orig).to(dev).permute(2, 0, 1)[None].float() / 255
    m = up_mask(torch.from_numpy(m8)[None].to(dev))
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        y = net(img, m).float()
        if a.tta:
            y = 0.5 * (y + net(img.flip(-1), m.flip(-1)).float().flip(-1))
    pred = (y.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    mpx = np.kron(m8, np.ones((32, 32), bool))[..., None]
    res = np.where(mpx, pred, orig)                       # visibles = originales exactos
    assert (res[~mpx[..., 0]] == orig[~mpx[..., 0]]).all()
    Image.fromarray(res).save(Path(a.out) / f"{r.ID}.png")
print("listo ->", a.out)
