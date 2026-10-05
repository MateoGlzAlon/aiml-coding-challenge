# One-off Kaggle script kernel: downloads EuroSAT_MS.zip, decodes every GeoTIFF with
# rasterio, and saves the whole training set as three files under /kaggle/working:
#   X_ms.npy (27000, 64, 64, 13) uint16, y.npy (27000,) int64, classes.json
#
# This mirrors Section 1 of small_track_colab.ipynb (same URL, same md5, same band
# layout), just pointed at Kaggle's filesystem instead of Google Drive. Kaggle's
# machines have more RAM than the team's laptops, which is the whole reason this
# build runs here instead of locally. Output becomes a private Kaggle Dataset
# (see scripts/make_cache_dataset.sh) that the training kernel attaches read-only,
# so this expensive step only has to happen once.
import hashlib
import json
import socket
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

socket.setdefaulttimeout(60)  # a stalled connection (no bytes for 60s) raises instead of hanging forever

# /kaggle/working is entirely snapshotted as the kernel's output, so the downloaded zip and
# the ~27,000 extracted GeoTIFFs (temporary, not needed afterwards) must live somewhere else -
# /kaggle/temp if it exists, else plain /tmp - or "output" balloons from ~2.7 GB to ~8 GB.
WORK_DIR = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else Path("/tmp/eurosat_scratch")
OUT_DIR = Path("/kaggle/working")
WORK_DIR.mkdir(parents=True, exist_ok=True)

EUROSAT_URL = "https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1"
EUROSAT_MD5 = "091174add3c8e680a49244acf185b9f0"

X_PATH, Y_PATH, CLASSES_PATH = OUT_DIR / "X_ms.npy", OUT_DIR / "y.npy", OUT_DIR / "classes.json"


def md5sum(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url, dest, attempts=3):
    last_pct = -1

    def progress(block_num, block_size, total_size):
        nonlocal last_pct
        done = block_num * block_size
        pct = int(100 * done / total_size) if total_size > 0 else 0
        if pct >= last_pct + 5:  # a line every ~5%, not one per 8KB block
            print(f"  {done / 1e6:.0f}/{total_size / 1e6:.0f} MB ({pct}%)")
            last_pct = pct

    for k in range(1, attempts + 1):
        try:
            urllib.request.urlretrieve(url, dest, reporthook=progress)
            return
        except Exception as e:
            print(f"download attempt {k} failed: {e}")
            time.sleep(10)
    raise RuntimeError("download failed - try again in a few minutes")


def read_tif(path):
    import rasterio as rio
    from rasterio.plot import reshape_as_image
    with rio.open(path, "r") as d:
        return reshape_as_image(d.read())


def main():
    try:
        import numpy as np
        import rasterio  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rasterio"], check=True)
        import numpy as np

    zip_path = WORK_DIR / "EuroSAT_MS.zip"
    if not zip_path.exists():
        print("downloading EuroSAT_MS.zip (2.1 GB) ...")
        download(EUROSAT_URL, zip_path)
    print("checking the download ...")
    if md5sum(zip_path) != EUROSAT_MD5:
        zip_path.unlink()
        raise RuntimeError("download corrupted (checksum mismatch) - run again")
    print("unpacking ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(WORK_DIR)
    root = WORK_DIR / "EuroSAT_MS"
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    files = [(f, i) for i, c in enumerate(classes) for f in sorted((root / c).glob("*.tif"))]
    first = read_tif(files[0][0])
    print(f"{len(files)} images, each {first.shape} {first.dtype}")
    X = np.empty((len(files),) + first.shape, dtype=first.dtype)
    for k, (f, _) in enumerate(files):
        X[k] = read_tif(f)
        if k % 5000 == 0:
            print(f"  decoded {k}/{len(files)}")
    y = np.array([i for _, i in files], dtype="int64")
    print("saving ...")
    np.save(X_PATH, X)
    np.save(Y_PATH, y)
    CLASSES_PATH.write_text(json.dumps(classes))
    print("done:", X_PATH, Y_PATH, CLASSES_PATH)


if __name__ == "__main__":
    main()
