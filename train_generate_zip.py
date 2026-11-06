"""Genera un zip de ~270 MB con todas las imágenes de train.

Los PNG casi no se comprimen dentro de un zip, así que se re-codifican como
JPEG (mismo tamaño 256x256, mismo ID/nombre) con una calidad única elegida por
búsqueda binaria sobre una muestra para que el zip final quede justo por
debajo del tamaño objetivo.

Uso:
    python train_generate_zip.py
    python train_generate_zip.py --target-mb 270 --output train_generate.zip
"""
import argparse
import io
import random
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

try:
    from tqdm import tqdm
except ImportError:  # tqdm es opcional
    def tqdm(it, **kwargs):
        return it

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ZIP_OVERHEAD_PER_FILE = 120  # cabeceras local + central directory (aprox.)


def encode(path: Path, quality: int, subsampling: int) -> bytes:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        buf = io.BytesIO()
        im.save(
            buf,
            format="JPEG",
            quality=quality,
            subsampling=subsampling,
            optimize=True,
        )
    return buf.getvalue()


def _encode_job(args):
    path, quality, subsampling = args
    return path.stem, encode(path, quality, subsampling)


def estimate_total(paths, sample, quality, subsampling) -> float:
    size = sum(len(encode(p, quality, subsampling)) for p in sample)
    return size / len(sample) * len(paths) + ZIP_OVERHEAD_PER_FILE * len(paths)


def find_quality(paths, target_bytes, subsampling, sample_size, seed) -> int:
    rng = random.Random(seed)
    sample = rng.sample(paths, min(sample_size, len(paths)))

    lo, hi, best = 1, 100, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        est = estimate_total(paths, sample, mid, subsampling)
        print(f"  calidad {mid:3d} -> estimado {est / 1e6:7.1f} MB")
        if est <= target_bytes:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def build_zip(paths, output, quality, subsampling, workers):
    jobs = [(p, quality, subsampling) for p in paths]
    with zipfile.ZipFile(output, "w", zipfile.ZIP_STORED) as zf, \
            ProcessPoolExecutor(max_workers=workers) as pool:
        for stem, data in tqdm(
            pool.map(_encode_job, jobs, chunksize=64),
            total=len(jobs),
            desc="Comprimiendo",
        ):
            zf.writestr(f"{stem}.jpg", data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", type=Path, default=Path("train"))
    ap.add_argument("--output", type=Path, default=Path("train_generate.zip"))
    ap.add_argument("--target-mb", type=float, default=270.0,
                    help="Tamaño máximo del zip en MB (1 MB = 1e6 bytes)")
    ap.add_argument("--subsampling", type=int, default=0, choices=[0, 1, 2],
                    help="0=4:4:4 (mejor color), 1=4:2:2, 2=4:2:0")
    ap.add_argument("--sample-size", type=int, default=400)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(
        p for p in args.images_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No hay imágenes en {args.images_root}")
    print(f"{len(paths)} imágenes en {args.images_root}")

    stems = [p.stem for p in paths]
    if len(set(stems)) != len(stems):
        raise ValueError("Hay IDs de imagen duplicados.")

    target_bytes = int(args.target_mb * 1e6)

    print("Buscando calidad JPEG...")
    quality = find_quality(
        paths, target_bytes, args.subsampling, args.sample_size, args.seed
    )
    print(f"Calidad elegida: {quality}")

    # Ajuste de seguridad: si el zip real se pasa, baja la calidad y repite.
    while True:
        build_zip(paths, args.output, quality, args.subsampling, args.workers)
        real = args.output.stat().st_size
        print(f"Zip: {args.output} -> {real / 1e6:.1f} MB (calidad {quality})")
        if real <= target_bytes or quality <= 1:
            break
        quality -= 1
        print(f"Se pasó del objetivo; reintentando con calidad {quality}")

    with zipfile.ZipFile(args.output) as zf:
        assert len(zf.namelist()) == len(paths), "faltan imágenes en el zip"
    print(f"OK: {len(paths)} imágenes, {real / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
