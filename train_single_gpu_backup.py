"""Entrenamiento de inpainting para accuracy de tokens DINOv3-VQ. Pensado para Kaggle:
presupuesto de tiempo (--time-budget-h), LR dependiente del tiempo, checkpoints atómicos y reanudación."""
import argparse, copy, math, os, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import DinoVQ, UNet, random_masks, up_mask

ap = argparse.ArgumentParser()
ap.add_argument("--train-dir", required=True)
ap.add_argument("--codebook", required=True)
ap.add_argument("--out", default="runs/v1")
ap.add_argument("--cache", default="", help="ruta .npy para cachear train en uint8")
ap.add_argument("--time-budget-h", type=float, default=9.5)
ap.add_argument("--max-steps", type=int, default=0, help="0 = sin límite (smoke test: p.ej. 50)")
ap.add_argument("--bs", type=int, default=24)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--warmup", type=int, default=500)
ap.add_argument("--ema", type=float, default=0.999)
ap.add_argument("--val-n", type=int, default=500)
ap.add_argument("--val-every", type=int, default=2000)
ap.add_argument("--save-every-min", type=float, default=15)
ap.add_argument("--tau", type=float, default=0.05)
ap.add_argument("--w-pix", type=float, default=1.0)
ap.add_argument("--w-cos", type=float, default=2.0)
ap.add_argument("--w-ce", type=float, default=0.5)
ap.add_argument("--resume", default="")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = "cuda"
torch.manual_seed(a.seed); np.random.seed(a.seed)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ datos (uint8 en RAM)
def load_train():
    if a.cache and Path(a.cache).exists():
        return np.load(a.cache, mmap_mode="r")
    files = sorted(p for p in Path(a.train_dir).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    print(len(files), "imágenes de train", flush=True)

    def rd(p):
        im = Image.open(p).convert("RGB")
        assert im.size == (256, 256), (p, im.size)
        return np.asarray(im)

    with ThreadPoolExecutor(8) as ex:
        arr = np.stack(list(ex.map(rd, files)))
    if a.cache:
        np.save(a.cache, arr)
    return arr


data = load_train()
N = len(data)
perm = np.random.default_rng(1234).permutation(N)          # split fijo independiente de --seed
val_idx, tr_idx = np.sort(perm[:a.val_n]), perm[a.val_n:]
val_imgs = torch.from_numpy(np.ascontiguousarray(data[val_idx]))
print(f"train={len(tr_idx)} val={len(val_idx)}", flush=True)

# ------------------------------------------------------------------ modelos
dino = DinoVQ(a.codebook, dev)
net = UNet().to(dev)
ema = copy.deepcopy(net).eval().requires_grad_(False)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda")
step, elapsed = 0, 0.0
print("parámetros U-Net: %.1fM" % (sum(p.numel() for p in net.parameters()) / 1e6), flush=True)

if a.resume and Path(a.resume).exists():
    ck = torch.load(a.resume, map_location=dev)
    net.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
    scaler.load_state_dict(ck["scaler"]); step, elapsed = ck["step"], ck["elapsed"]
    print(f"reanudado: step={step} tiempo previo={elapsed/3600:.2f}h", flush=True)


def save(path):
    tmp = str(path) + ".tmp"
    torch.save({"model": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "step": step, "elapsed": elapsed}, tmp)
    os.replace(tmp, path)


# ------------------------------------------------------------------ validación (misma cadena que la submission)
@torch.no_grad()
def validate(model, black_baseline=False):
    g = torch.Generator(device=dev); g.manual_seed(999)
    hit = tot = 0
    for i in range(0, len(val_imgs), 25):
        img = val_imgs[i:i + 25].to(dev).permute(0, 3, 1, 2).float() / 255
        m8 = random_masks(len(img), dev, g); m = up_mask(m8)
        with torch.autocast("cuda", torch.float16):
            gt = dino.codes(img)
            pred = img * (1 - m) if black_baseline else model(img, m)
            pred = (pred.float().clamp(0, 1) * 255).round() / 255          # redondeo a uint8 como al guardar PNG
            pc = dino.codes(pred)
        hit += ((pc == gt) & m8.view(-1, 64)).sum().item(); tot += m8.sum().item()
    return hit / tot


base = validate(None, True)
print(f"baseline parche negro: VAL acc={base:.4f}", flush=True)

# ------------------------------------------------------------------ entrenamiento
budget = a.time_budget_h * 3600
ema_params, net_params = list(ema.parameters()), list(net.parameters())
tr_t = torch.from_numpy(np.ascontiguousarray(data[tr_idx]))   # ~3 GB en RAM
order, ptr = np.random.default_rng(a.seed + step).permutation(len(tr_idx)), 0
t0 = time.time() - elapsed
last_save = time.time(); run = {"loss": 0, "pix": 0, "cos": 0, "ce": 0, "n": 0}
net.train()

while True:
    elapsed = time.time() - t0
    if elapsed >= budget or (a.max_steps and step >= a.max_steps):
        break
    if ptr + a.bs > len(order):
        order, ptr = np.random.default_rng(a.seed + step).permutation(len(tr_idx)), 0
    idx = torch.from_numpy(order[ptr:ptr + a.bs]); ptr += a.bs
    img = tr_t[idx].to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255
    fh = torch.rand(len(img), device=dev) < 0.5
    img = torch.where(fh[:, None, None, None], img.flip(-1), img)           # flip horizontal aleatorio
    m8 = random_masks(len(img), dev); m = up_mask(m8); mt = m8.view(len(img), 64)

    lr = a.lr * min(1, (step + 1) / a.warmup) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1, elapsed / budget))))
    for gp in opt.param_groups:
        gp["lr"] = lr

    with torch.autocast("cuda", torch.float16):
        pred = net(img, m)
        e = dino.embed(pred)
        with torch.no_grad():
            e_gt = dino.embed(img)
            tgt = (e_gt @ dino.codebook.T).argmax(-1)
    pix = ((pred.float() - img).abs() * m).sum() / (m.sum() * 3).clamp(min=1)
    cos = (1 - (e * e_gt).sum(-1))[mt].mean()
    ce = F.cross_entropy((e @ dino.codebook.T)[mt] / a.tau, tgt[mt])
    loss = a.w_pix * pix + a.w_cos * cos + a.w_ce * ce

    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    scaler.step(opt); scaler.update()
    d = min(a.ema, (1 + step) / (10 + step))
    torch._foreach_mul_(ema_params, d); torch._foreach_add_(ema_params, [p.detach() for p in net_params], alpha=1 - d)
    step += 1

    for k, v in (("loss", loss), ("pix", pix), ("cos", cos), ("ce", ce)):
        run[k] += v.item() if step % 20 == 0 else 0
    run["n"] += step % 20 == 0
    if step % 200 == 0 and run["n"]:
        n = run["n"]
        print(f"step {step} t={elapsed/3600:.2f}h lr={lr:.2e} loss={run['loss']/n:.4f} pix={run['pix']/n:.4f} "
              f"cos={run['cos']/n:.4f} ce={run['ce']/n:.3f}", flush=True)
        run = {k: 0 for k in run}
    if step % a.val_every == 0:
        print(f"== step {step} VAL acc (EMA)={validate(ema):.4f} (baseline negro={base:.4f})", flush=True)
    if time.time() - last_save > a.save_every_min * 60:
        save(out / "last.pt"); last_save = time.time()

elapsed = time.time() - t0
save(out / "last.pt")
print(f"FIN step={step} t={elapsed/3600:.2f}h | VAL acc (EMA)={validate(ema):.4f} (baseline negro={base:.4f})", flush=True)
