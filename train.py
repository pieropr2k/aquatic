"""Entrenamiento de inpainting para accuracy de tokens DINOv3-VQ. Pensado para Kaggle:
presupuesto de tiempo (--time-budget-h), LR dependiente del tiempo, checkpoints atómicos y reanudación."""
import argparse, copy, math, os, time
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image

from common import DinoVQ, UNet, mask_from_target, random_masks, up_mask

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
ap.add_argument("--channels-last", type=int, default=0, help="1 = U-Net en channels_last (probar si acelera)")
ap.add_argument("--gt-bs", type=int, default=64, help="batch para precomputar embeddings GT")
ap.add_argument("--mask-csv", default="", help="target.csv: usar sus máscaras reales (banco) en train y val")
ap.add_argument("--p-real", type=float, default=0.5, help="prob. de usar una máscara del banco en train (resto: uniforme)")
ap.add_argument("--group-split", type=int, default=1, help="1 = val por clusters DINO (sin imágenes casi duplicadas en train)")
ap.add_argument("--n-clusters", type=int, default=400)
ap.add_argument("--photo-aug", type=float, default=0.0, help="prob. de augmentación de brillo/contraste/color (el test es más brillante y con más contraste que train)")
a = ap.parse_args()

# ------------------------------------------------------------------ multi-GPU (torchrun) o 1 GPU (python)
# --bs es POR GPU: batch efectivo = bs * world.
rank, world, local = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local)
if world > 1:
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
dev = f"cuda:{local}"


def barrier():
    if world > 1:
        dist.barrier()


def P(*args, **kw):
    if rank == 0:
        print(*args, **kw, flush=True)


torch.manual_seed(a.seed); np.random.seed(a.seed)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ máscaras: uniformes + banco de máscaras reales del test
bank = None
if a.mask_csv:
    import csv
    with open(a.mask_csv, encoding="utf-8") as f:
        bank = torch.from_numpy(np.stack([mask_from_target(int(x) for x in r["target"].split()) for r in csv.DictReader(f)])).to(dev)
    P(f"banco de máscaras reales: {len(bank)} (p_real train={a.p_real})")


def photo_aug(img, p, c=None, g=None):
    """Brillo/contraste/color aleatorios sobre (B,3,H,W) en [0,1]. c,g fijos -> shift determinista (para validar)."""
    B = len(img)
    if c is None:
        c = torch.exp(torch.empty(B, 1, 1, 1, device=img.device).uniform_(math.log(0.8), math.log(2.5)))   # contraste (test ~1.8x train)
        g = torch.exp(torch.empty(B, 1, 1, 1, device=img.device).uniform_(math.log(0.8), math.log(1.8)))   # ganancia de brillo
        ch = 1 + 0.08 * torch.randn(B, 3, 1, 1, device=img.device)                                           # leve cambio de color
    else:
        ch = 1.0
    mu = img.mean((1, 2, 3), keepdim=True)
    out = (((img - mu) * c + mu) * g * ch).clamp(0, 1)
    out = (out * 255).round() / 255
    use = torch.rand(B, 1, 1, 1, device=img.device) < p
    return torch.where(use, out, img)


def sample_masks(B, gen=None, p_real=0.0, aug=True):
    """(B,8,8) bool. Con prob. p_real toma una máscara real del banco (flip horizontal opcional), si no una uniforme."""
    m = random_masks(B, dev, gen)
    if bank is None or p_real <= 0:
        return m
    pick = bank[torch.randint(len(bank), (B,), device=dev, generator=gen)]
    if aug:
        pick = torch.where((torch.rand(B, device=dev, generator=gen) < 0.5)[:, None, None], pick.flip(-1), pick)
    use = torch.rand(B, device=dev, generator=gen) < p_real
    return torch.where(use[:, None, None], pick, m)


# ------------------------------------------------------------------ datos (uint8 en RAM)
def load_train():
    if a.cache and Path(a.cache).exists():
        return np.load(a.cache, mmap_mode="r")
    files = sorted(p for p in Path(a.train_dir).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    P(len(files), "imágenes de train")

    def rd(p):
        im = Image.open(p).convert("RGB")
        assert im.size == (256, 256), (p, im.size)
        return np.asarray(im)

    with ThreadPoolExecutor(8) as ex:
        arr = np.stack(list(ex.map(rd, files)))
    if a.cache:
        np.save(a.cache, arr)
    return arr


if rank == 0:                     # rank 0 crea la caché .npy, el resto la lee
    data = load_train()
barrier()
if rank != 0:
    data = load_train()
N = len(data)       # el split train/val se hace más abajo (el modo por clusters necesita los embeddings GT)

# ------------------------------------------------------------------ modelos
if rank == 0:                     # rank 0 descarga/carga primero (evita descargas simultáneas)
    dino = DinoVQ(a.codebook, dev)
barrier()
if rank != 0:
    dino = DinoVQ(a.codebook, dev)
torch.manual_seed(a.seed)         # misma init de la U-Net en todos los ranks (y por tanto del EMA)
net = UNet().to(dev)
if a.channels_last:
    net = net.to(memory_format=torch.channels_last)
torch.manual_seed(a.seed + rank)  # máscaras/flips distintos por rank
ema = copy.deepcopy(net).eval().requires_grad_(False)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda")
step, elapsed = 0, 0.0
P("parámetros U-Net: %.1fM" % (sum(p.numel() for p in net.parameters()) / 1e6))

if a.resume and Path(a.resume).exists():
    ck = torch.load(a.resume, map_location=dev)
    net.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
    scaler.load_state_dict(ck["scaler"]); step, elapsed = ck["step"], ck["elapsed"]
    P(f"reanudado: step={step} tiempo previo={elapsed/3600:.2f}h")

net_fwd = DDP(net, device_ids=[local]) if world > 1 else net


def save(path):
    tmp = str(path) + ".tmp"
    torch.save({"model": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "step": step, "elapsed": elapsed}, tmp)
    os.replace(tmp, path)


# ------------------------------------------------------------------ validación (misma cadena que la submission)
@torch.no_grad()
def validate(model, black_baseline=False, real=True, shift=False):
    """real=True: máscaras del banco (como el test) si hay --mask-csv; real=False: uniformes.
    shift=True: val con más contraste y brillo (imita al test)."""
    g = torch.Generator(device=dev); g.manual_seed(999)
    hit = tot = 0
    for i in range(0, len(val_imgs), 25):
        img = val_imgs[i:i + 25].to(dev).permute(0, 3, 1, 2).float() / 255
        if shift:
            img = photo_aug(img, 1.0, c=1.8, g=1.25)
        m8 = sample_masks(len(img), g, 1.0 if real else 0.0, aug=False); m = up_mask(m8)
        with torch.autocast("cuda", torch.float16):
            gt = dino.codes(img)
            pred = img * (1 - m) if black_baseline else model(img, m)
            pred = (pred.float().clamp(0, 1) * 255).round() / 255          # redondeo a uint8 como al guardar PNG
            pc = dino.codes(pred)
        hit += ((pc == gt) & m8.view(-1, 64)).sum().item(); tot += m8.sum().item()
    return hit / tot


# ------------------------------------------------------------------ embeddings GT precomputados (no dependen de la máscara)
@torch.no_grad()
def build_gt():
    """(N,2,64,D) fp16: embedding DINO de cada imagen sin flip / con flip. Se reparte entre GPUs y se cachea en disco."""
    path = Path(a.cache + ".gt.npy") if a.cache else out / "gt.npy"
    done = Path(str(path) + ".done")
    if not done.exists():
        D = dino.codebook.shape[1]
        if rank == 0:
            mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(N, 2, 64, D)); del mm
        barrier()
        mm = np.load(path, mmap_mode="r+")
        t = time.time()
        for s in range(rank * a.gt_bs, N, world * a.gt_bs):
            e = min(s + a.gt_bs, N)
            img = torch.from_numpy(np.ascontiguousarray(data[s:e])).to(dev).permute(0, 3, 1, 2).float() / 255
            with torch.autocast("cuda", torch.float16):
                e0, e1 = dino.embed(img), dino.embed(img.flip(-1))
            mm[s:e, 0] = e0.half().cpu().numpy(); mm[s:e, 1] = e1.half().cpu().numpy()
        mm.flush(); del mm
        barrier()
        if rank == 0:
            done.touch()
        barrier()
        P(f"embeddings GT precomputados en {time.time() - t:.0f}s -> {path}")
    return np.load(path, mmap_mode="r")


gt_all = build_gt()


# ------------------------------------------------------------------ split train/val
def kmeans_cos(x, k, iters=25):
    x = torch.from_numpy(x).to(dev)
    g = torch.Generator(device=dev); g.manual_seed(0)
    c = x[torch.randperm(len(x), device=dev, generator=g)[:k]].clone()
    for _ in range(iters):
        lab = (x @ c.T).argmax(1)
        s = torch.zeros_like(c).index_add_(0, lab, x)
        nz = torch.bincount(lab, minlength=k) > 0
        c[nz] = F.normalize(s[nz], dim=-1)
    return (x @ c.T).argmax(1).cpu().numpy()


def make_split():
    sp = out / f"split_{a.val_n}_{'group' if a.group_split else 'rand'}.npz"
    if not sp.exists() and rank == 0:
        rng = np.random.default_rng(1234)                                  # independiente de --seed
        if a.group_split:
            gv = np.concatenate([F.normalize(torch.from_numpy(np.asarray(gt_all[i:i + 2048, 0], dtype=np.float32)).mean(1), dim=-1).numpy()
                                 for i in range(0, N, 2048)])               # vector global por imagen (media de 64 parches)
            lab = kmeans_cos(gv, a.n_clusters)
            val = []
            for c in rng.permutation(a.n_clusters):                         # clusters completos -> val
                idx = np.where(lab == c)[0]
                if len(idx) and len(val) + len(idx) <= 1.5 * a.val_n:
                    val += idx.tolist()
                if len(val) >= a.val_n:
                    break
            val_idx = np.sort(np.array(val)); tr_idx = np.setdiff1d(np.arange(N), val_idx)
            ms = lambda A, B: (torch.from_numpy(gv[A]).to(dev) @ torch.from_numpy(gv[B]).to(dev).T).max(1).values.mean().item()
            ridx = rng.permutation(N)[:len(val_idx)]
            print(f"similitud máx. val->train: grupos={ms(val_idx, tr_idx):.3f} | split aleatorio de referencia={ms(ridx, np.setdiff1d(np.arange(N), ridx)):.3f}", flush=True)
        else:
            perm = rng.permutation(N); val_idx, tr_idx = np.sort(perm[:a.val_n]), perm[a.val_n:]
        np.savez(sp, val=val_idx, tr=tr_idx)
    barrier()
    z = np.load(sp)
    return z["val"], z["tr"]


val_idx, tr_idx = make_split()
val_imgs = torch.from_numpy(np.ascontiguousarray(data[val_idx]))
P(f"train={len(tr_idx)} val={len(val_idx)} | GPUs={world} bs/GPU={a.bs} bs efectivo={a.bs * world}")
base = validate(None, True) if rank == 0 else 0.0
P(f"baseline parche negro: VAL acc={base:.4f}")

# ------------------------------------------------------------------ entrenamiento
budget = a.time_budget_h * 3600
ema_params, net_params = list(ema.parameters()), list(net.parameters())
tr_t = torch.from_numpy(np.ascontiguousarray(data[tr_idx]))   # ~3 GB en RAM
order, ptr = np.random.default_rng(a.seed + step).permutation(len(tr_idx))[rank::world], 0   # shards disjuntos por rank
t0 = time.time() - elapsed
last_save = time.time(); run = {"loss": 0, "pix": 0, "cos": 0, "ce": 0, "n": 0}
t_log, s_log = time.time(), step
net.train()

while True:
    tt = torch.tensor([time.time() - t0], device=dev, dtype=torch.float64)
    if world > 1:
        dist.broadcast(tt, 0)         # todos los ranks usan el mismo reloj (corte y LR sincronizados)
    elapsed = tt.item()
    if elapsed >= budget or (a.max_steps and step >= a.max_steps):
        break
    if ptr + a.bs > len(order):
        order, ptr = np.random.default_rng(a.seed + step).permutation(len(tr_idx))[rank::world], 0
    idx = torch.from_numpy(order[ptr:ptr + a.bs]); ptr += a.bs
    img = tr_t[idx].to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255
    fh_c = torch.rand(len(img)) < 0.5
    fh = fh_c.to(dev)
    img = torch.where(fh[:, None, None, None], img.flip(-1), img)           # flip horizontal aleatorio
    if a.photo_aug > 0:
        img = photo_aug(img, a.photo_aug)
    m8 = sample_masks(len(img), None, a.p_real); m = up_mask(m8); mt = m8.view(len(img), 64)

    lr = a.lr * min(1, (step + 1) / a.warmup) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1, elapsed / budget))))
    for gp in opt.param_groups:
        gp["lr"] = lr

    if a.photo_aug > 0:               # con augmentación fotométrica el GT cacheado ya no vale: recalcular
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            e_gt = dino.embed(img).float()
    else:
        e_gt = torch.from_numpy(gt_all[tr_idx[idx.numpy()], fh_c.numpy().astype(np.int64)]).to(dev).float()   # GT precomputado
    tgt =(e_gt @ dino.codebook.T).argmax(-1)
    with torch.autocast("cuda", torch.float16):
        pred = net_fwd(img, m)
        e = dino.embed(pred)
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
        P(f"step {step} t={elapsed/3600:.2f}h lr={lr:.2e} loss={run['loss']/n:.4f} pix={run['pix']/n:.4f} "
          f"cos={run['cos']/n:.4f} ce={run['ce']/n:.3f} | {(step - s_log) / (time.time() - t_log):.2f} it/s")
        run = {k: 0 for k in run}; t_log, s_log = time.time(), step
    if step % a.val_every == 0 and rank == 0:
        print(f"== step {step} VAL acc (EMA)={validate(ema):.4f} | VAL-shift (más contraste/brillo, como el test)={validate(ema, shift=True):.4f} "
              f"(baseline negro={base:.4f})", flush=True)
    if rank == 0 and time.time() - last_save > a.save_every_min * 60:
        save(out / "last.pt"); last_save = time.time()

barrier()
if rank == 0:
    save(out / "last.pt")
    print(f"FIN step={step} t={elapsed/3600:.2f}h | VAL acc (EMA)={validate(ema):.4f} (baseline negro={base:.4f})", flush=True)
    if bank is not None:
        print(f"   VAL con máscaras uniformes: {validate(ema, real=False):.4f} | VAL-shift: {validate(ema, shift=True):.4f}", flush=True)
barrier()
if world > 1:
    dist.destroy_process_group()
