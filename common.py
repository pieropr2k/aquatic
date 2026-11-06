"""Utilidades compartidas: máscaras, DINOv3-VQ diferenciable (misma cadena que dino_vq.py) y modelo U-Net."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
LAYER = 10


# ------------------------------------------------------------------ máscaras
def random_masks(B, device, gen=None):
    """(B,8,8) bool; k~U[8,20] patches ocultos por imagen, posiciones uniformes (como en test)."""
    k = torch.randint(8, 21, (B, 1), device=device, generator=gen)
    scores = torch.rand(B, 64, device=device, generator=gen)
    ranks = scores.argsort(1).argsort(1)
    return (ranks < k).view(B, 8, 8)


def mask_from_target(patches):
    m = np.zeros(64, bool)
    m[list(patches)] = True
    return m.reshape(8, 8)


def up_mask(m8):
    """(B,8,8) bool -> (B,1,256,256) float."""
    return F.interpolate(m8[:, None].float(), scale_factor=32, mode="nearest")


# ------------------------------------------------------------------ DINOv3 + codebook
class DinoVQ(nn.Module):
    """Réplica diferenciable de dino_vq.DinoVQ. Entrada en [0,1], (B,3,256,256)."""

    def __init__(self, codebook_path, device, truncate=True):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor(proc.image_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(proc.image_std).view(1, 3, 1, 1))
        cb = torch.from_numpy(np.load(codebook_path).astype(np.float32))
        self.register_buffer("codebook", F.normalize(cb, dim=1))
        self.to(device)
        if truncate and hasattr(self.model, "layer"):      # hidden_states[10] solo necesita 10 bloques
            x = torch.rand(2, 3, 256, 256, device=device)
            with torch.no_grad():
                ref = self.embed(x)
                full = self.model.layer
                self.model.layer = full[:LAYER]
                ok = torch.allclose(ref, self.embed(x), atol=1e-3)
            if not ok:                                      # por seguridad, volver al modelo completo
                self.model.layer = full
            print("DINO truncado a", LAYER, "capas" if ok else "-> truncado desactivado", flush=True)

    def embed(self, x01):
        h = self.model(pixel_values=(x01 - self.mean) / self.std, output_hidden_states=True).hidden_states[LAYER]
        sp = h[:, -256:].reshape(-1, 16, 16, h.shape[-1])
        B, D = sp.shape[0], sp.shape[-1]
        p = sp.reshape(B, 8, 2, 8, 2, D).mean(dim=(2, 4))
        return F.normalize(p.float(), dim=-1).reshape(B, 64, D)

    def codes(self, x01):
        return (self.embed(x01) @ self.codebook.T).argmax(-1)       # (B,64)


# ------------------------------------------------------------------ modelo
def cbn(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.SiLU())


class Res(nn.Module):
    def __init__(self, i, o, s=1):
        super().__init__()
        self.a = cbn(i, o, s)
        self.b = nn.Sequential(nn.Conv2d(o, o, 3, 1, 1), nn.GroupNorm(8, o))
        self.sk = nn.Identity() if (i == o and s == 1) else nn.Conv2d(i, o, 1, s)

    def forward(self, x):
        return F.silu(self.b(self.a(x)) + self.sk(x))


class UNet(nn.Module):
    """Entrada: imagen enmascarada + máscara (4 canales); salida RGB [0,1].
    Cuello 16x16 con transformer (contexto global). Salida compuesta: los píxeles visibles no cambian."""

    def __init__(self, ch=(64, 128, 256, 384, 512), tf_layers=6):
        super().__init__()
        self.inp = cbn(4, ch[0])
        self.enc = nn.ModuleList([Res(ch[0], ch[0])] + [Res(ch[i - 1], ch[i], 2) for i in range(1, 5)])  # 256..16
        d = ch[4]
        self.pos = nn.Parameter(torch.zeros(1, 256, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, 8, d * 4, 0.0, "gelu", batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, tf_layers, enable_nested_tensor=False)
        self.dec = nn.ModuleList([Res(ch[i] + ch[i - 1], ch[i - 1]) for i in range(4, 0, -1)])
        self.out = nn.Conv2d(ch[0] * 2, 3, 3, 1, 1)

    def forward(self, img, m):                      # img (B,3,H,W) en [0,1]; m (B,1,H,W) 1=oculto
        x = self.inp(torch.cat([img * (1 - m), m], 1))
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
        B, C, H, W = x.shape
        x = self.tf(x.flatten(2).transpose(1, 2) + self.pos).transpose(1, 2).reshape(B, C, H, W)
        for i, dcd in enumerate(self.dec):
            sk = skips[-2 - i]
            x = F.interpolate(x, size=sk.shape[-2:], mode="bilinear", align_corners=False)
            x = dcd(torch.cat([x, sk], 1))
        y = torch.sigmoid(self.out(torch.cat([x, skips[0]], 1)).float())
        return m * y + (1 - m) * img
