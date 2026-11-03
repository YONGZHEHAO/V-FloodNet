"""Download official weights during setup; never publish the binary files to Git.

The records archive is pinned to a Hugging Face revision and LFS SHA256.
Detectron2's official model URL is fixed; an expected SHA256 is recorded below.
"""
import argparse
import hashlib
from pathlib import Path
import os
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = "a0b9c2de193de188434220949f10da84e61844cb"
RECORDS_URL = f"https://huggingface.co/xmlyqing00/V-FloodNet/resolve/{REVISION}/records.zip"
RECORDS_SHA256 = "fa2952ff789906bf4cfa13e415a59bb38da49018cfb781bd4905af9077156f01"
STOPSIGN_URL = ("https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/"
                "pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl")
# Filled from the downloaded official file; do not change without checking provenance.
STOPSIGN_SHA256 = "23c63e468274851e212c6d7525336e394903c5e290f31557f7150d83b2a52b44"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, destination, expected_sha256):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256(destination) == expected_sha256:
        print(f"Verified cached file: {destination}", flush=True)
        return
    temporary = destination.with_name(destination.name + ".part")
    print(f"Downloading {url} -> {destination}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "V-FloodNet-Cloud-Setup"})
    try:
        with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
        actual = sha256(temporary)
        if actual != expected_sha256:
            raise RuntimeError(f"SHA256 mismatch for {destination.name}: {actual}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("cpu", "image"), default="cpu")
    args = parser.parse_args()
    cache = Path(os.environ.get("VFLOOD_CACHE_DIR", ROOT / ".codex/cache/models"))
    archive = cache / "records.zip"
    download(RECORDS_URL, archive, RECORDS_SHA256)
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "records":
                raise RuntimeError(f"Unexpected archive path: {member.filename}")
            # Keep image weights and calibration metadata. The GPU video model
            # is intentionally not extracted into the default CPU environment.
            if path.suffix == ".pth" and path.name != "link_efficientb4_model.pth":
                continue
            if not member.is_dir():
                target = ROOT / path
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    if args.profile == "cpu":
        download(STOPSIGN_URL, ROOT / "records/detectron2/model_final_ba17b9.pkl", STOPSIGN_SHA256)
    print("Official weights prepared. SMPL and human mesh models are separate requirements.")


if __name__ == "__main__":
    main()
