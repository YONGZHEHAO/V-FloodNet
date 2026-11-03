"""Check real CPU operators, imports, weights and optional official-model inference."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("cpu", "image"), default="cpu")
    parser.add_argument("--require-weights", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Segment one bundled photo with official weights.")
    parser.add_argument("--depth-smoke", action="store_true", help="Run the stop sign pipeline on one bundled lake photo (CPU profile only).")
    args = parser.parse_args()
    if args.depth_smoke and args.profile != "cpu":
        parser.error("--depth-smoke requires --profile cpu")
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    import torchvision
    import segmentation_models_pytorch
    import myutils
    from test_image_seg import test_waterseg

    torch.set_num_threads(2)
    assert torch.__version__ == "2.2.2+cpu", torch.__version__
    assert torchvision.__version__ == "0.17.2+cpu", torchvision.__version__
    selected = torchvision.ops.nms(torch.tensor([[0., 0., 2., 2.]]), torch.ones(1), 0.5)
    assert selected.tolist() == [0], "torchvision C++ operator failed"
    _, labels = cv2.connectedComponents(np.zeros((8, 8), dtype=np.uint8))
    assert labels.shape == (8, 8)
    assert hasattr(cv2, "legacy") and hasattr(cv2.legacy, "MultiTracker_create")

    image_weights = ROOT / "records/link_efficientb4_model.pth"
    detection_weights = ROOT / "records/detectron2/model_final_ba17b9.pkl"
    report = {
        "profile": args.profile,
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "opencv": cv2.__version__,
        "segmentation_models_pytorch": segmentation_models_pytorch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "image_weights_present": image_weights.is_file(),
        "stopsign_weights_present": detection_weights.is_file(),
        "human_mesh_configured": False,
        "video_cuda_configured": False,
        "smoke_inference": "not run",
        "stopsign_inference": "not run",
    }
    if args.profile == "cpu":
        from detectron2 import _C
        from detectron2.config import get_cfg
        from detectron2.projects.point_rend import add_pointrend_config
        import estimation.object_detection
        cfg = get_cfg()
        add_pointrend_config(cfg)
        cfg.merge_from_file(str(ROOT / "estimation/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml"))
        report["detectron2"] = importlib.metadata.version("detectron2")
        report["detectron2_cpp_loaded"] = bool(_C)
    if args.require_weights or args.smoke or args.depth_smoke:
        assert image_weights.is_file(), "Missing image weights; run scripts/download_cloud_weights.py"
        if args.profile == "cpu":
            assert detection_weights.is_file(), "Missing stop sign detector weights"
    if args.smoke:
        sample = sorted((ROOT / "assets/img_exp").glob("*.jpg"))[0]
        test_waterseg(str(image_weights), str(sample), "cloud_smoke", str(ROOT / "output/segs"), torch.device("cpu"))
        mask_path = ROOT / "output/segs/cloud_smoke/mask" / (sample.stem + ".png")
        with Image.open(mask_path) as mask, Image.open(sample) as original:
            assert mask.size == original.size
            assert set(np.unique(np.array(mask))).issubset({0, 1})
        report["smoke_inference"] = "passed (official model, real bundled image, CPU)"
        report["smoke_mask"] = str(mask_path.relative_to(ROOT))
    if args.depth_smoke:
        sample = ROOT / "assets/lake_exp/0.jpg"
        (ROOT / "output").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="depth-smoke-", dir=ROOT / "output") as folder:
            image_path = Path(folder) / sample.name
            shutil.copyfile(sample, image_path)
            test_waterseg(str(image_weights), str(image_path), "cloud_stopsign", str(ROOT / "output/segs"), torch.device("cpu"))
            subprocess.run([
                sys.executable, "est_waterlevel.py", "--test-path", folder,
                "--test-name", "cloud_stopsign", "--opt", "stopsign", "--device", "cpu",
            ], cwd=ROOT, check=True)
        result = ROOT / "output/waterlevel/cloud_stopsign_stopsign/waterdepth.txt"
        columns = result.read_text().strip().split()
        assert len(columns) == 3 and 0 <= float(columns[2]) <= 215.9, result.read_text()
        report["stopsign_inference"] = "passed (official detector, bundled photo, upstream reference dimensions; accuracy unvalidated)"
        report["stopsign_result"] = str(result.relative_to(ROOT))
        report["stopsign_demo_depth_cm"] = float(columns[2])
    destination = ROOT / ".codex/environment-report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
