import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download_coco(data_dir: str = str(PROJECT_ROOT / "data/coco")) -> dict:
    """
    Download COCO val2017 images and annotations if not present.

    Returns:
        {"images_dir": str, "annotations_path": str}
    """
    root = Path(data_dir)
    images_dir = root / "val2017"
    ann_path = root / "annotations" / "instances_val2017.json"
    root.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        print("Downloading COCO val2017 images (~1GB)...")
        _download_and_extract(COCO_IMAGES_URL, root)

    if not ann_path.exists():
        print("Downloading COCO annotations (~241MB)...")
        _download_and_extract(COCO_ANNOTATIONS_URL, root)

    return {"images_dir": str(images_dir), "annotations_path": str(ann_path)}


def _download_and_extract(url: str, dest: Path):
    zip_path = dest / url.split("/")[-1]
    print(f"  {url}")
    urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
    print()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    zip_path.unlink()
    print(f"  Extracted to {dest}")


def _progress(blocks, block_size, total):
    if total <= 0:
        return
    pct = min(blocks * block_size / total * 100, 100)
    print(f"\r  {pct:.1f}%", end="", flush=True)
