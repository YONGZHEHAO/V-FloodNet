"""Download the public METRO checkpoint and HRNet architecture configuration.

The licensed SMPL file is obtained by the user from the official website.
The METRO checkpoint includes backbone weights, so a second HRNet checkpoint
is not needed when every key is loaded strictly.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from download_cloud_weights import REVISION, download

BASE = f"https://huggingface.co/xmlyqing00/V-FloodNet/resolve/{REVISION}"
CHECKPOINT_SHA = "4d0c9808ec3bf315b027d2d38fd07ca8bf506afc0fda326275819d836d260aef"
CONFIG_SHA = "4bed04801a66f47c0a54604f46181d609d5d75c1588556d80ca77f380356301c"


def main():
    download(BASE + "/MeshTransformer/models/metro_release/metro_3dpw_state_dict.bin",
             ROOT / "records/people/metro_3dpw_state_dict.bin", CHECKPOINT_SHA)
    download(BASE + "/MeshTransformer/models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml",
             ROOT / "records/people/hrnet_w64.yaml", CONFIG_SHA)
    print("Public METRO weights verified. SMPL is a separate user-supplied file.")


if __name__ == "__main__":
    main()
