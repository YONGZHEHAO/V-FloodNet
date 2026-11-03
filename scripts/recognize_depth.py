"""Run water segmentation and reference-object screening in one CPU command.

Only a successful stop sign geometry fit with explicitly supplied reference
dimensions produces an estimated depth. Image masks and detections are retained
even when metric depth is unavailable. No source depth/ground-truth is updated.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import quote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STATUS = {
    "no_stop_sign": "未检出停止标志；当前路线无法估深",
    "reference_dimensions_missing": "检出停止标志候选；缺少已知尺寸，水深留空",
    "reference_geometry_failed": "停止标志几何拟合失败；水深留空",
    "estimated_unvalidated": "已按提供尺寸估计；尚未实测验证",
    "demo_assumptions": "演示估计；使用上游默认尺寸，尚未实测验证",
    "no_water_pixels": "模型掩膜无水像素；未作水深估计",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_results(output, report):
    (output / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cards = []
    for row in report["images"]:
        source_links = []
        for number, url in enumerate(row.get("source_urls", []), 1):
            if urlparse(url).scheme in {"http", "https"}:
                source_links.append(f'<a href="{html.escape(url, quote=True)}">来源{number}</a>')
        paths = [("overlay", "水体分割"), ("detections", "参照物检测")]
        figures = []
        for folder, label in paths:
            link = f'{folder}/{quote(row["image_id"])}.png'
            figures.append(f'<a href="{link}"><img loading="lazy" src="{link}" alt="{label}"></a>')
        depth = row["estimated_depth_cm"]
        depth_text = "无法给出厘米水深" if depth is None else f'估计 {depth:.2f} cm（未验证）'
        counts = row["reference_counts"]
        cards.append(
            f'<article><h2>{html.escape(row["image_id"])}</h2>'
            f'<p><strong>{depth_text}</strong> · {STATUS[row["depth_status"]]}</p>'
            f'<p>模型水体占图像 {row["model_water_fraction"]:.1%}；'
            f'人物 {counts["person"]}、车辆 {counts["vehicle"]}、停止标志 {counts["stop_sign"]}（模型检测，待人工复核）</p>'
            f'<div class="images">{"".join(figures)}</div>'
            f'<p>{" · ".join(source_links)}</p></article>')
    body = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>V-FloodNet 识别结果</title><style>'
        'body{font:16px/1.6 system-ui,sans-serif;margin:32px auto;padding:0 24px;max-width:1200px;background:#f4f6f8;color:#172332}'
        'article{background:white;padding:20px;margin:20px 0;border:1px solid #dde3e9;border-radius:10px}'
        'h2{font-size:16px;overflow-wrap:anywhere}.images{display:flex;gap:12px;flex-wrap:wrap}'
        '.images a{flex:1;min-width:240px}.images img{width:100%;max-height:500px;object-fit:contain;background:#e9eef2}'
        'a{color:#09618b}strong{color:#8c3d00}</style><h1>V-FloodNet 识别结果</h1>'
        f'<p>已处理 {len(report["images"])} 张；有数值的模型估计 {report["summary"]["numeric_estimates"]} 张；'
        '缺少条件时水深为 null，不填零、不套用人物身高或车辆尺寸。</p>'
        '<p>这些是模型推理结果。水体掩膜、检测框和来源事件关联都不是真值。'
        '播放器界面、夜景、遮挡与参考物尺寸会影响判断，需结合原图人工复核。</p>'
        '<p><a href="results.json">完整结果与来源记录</a></p>' + "".join(cards) + '</html>')
    (output / "index.html").write_text(body, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--test-name", required=True)
    parser.add_argument("--manifest", type=Path, help="Optional source manifest with images/file and sha256 fields.")
    parser.add_argument("--sign-size-cm", type=float)
    parser.add_argument("--pole-height-cm", type=float)
    parser.add_argument("--demo", action="store_true", help="Use upstream dimensions only for the bundled assets/lake_exp demo.")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.test_name):
        parser.error("test-name must contain only ASCII letters, digits, underscores or hyphens")
    dimensions = (args.sign_size_cm, args.pole_height_cm)
    if args.demo:
        if args.input_dir.resolve() != (ROOT / "assets/lake_exp").resolve() or any(x is not None for x in dimensions):
            parser.error("--demo is only for assets/lake_exp and cannot be combined with supplied dimensions")
        dimensions = (79.0, 215.9)
    elif any(x is not None for x in dimensions):
        if not all(x is not None and math.isfinite(x) and x > 0 for x in dimensions):
            parser.error("Supply both positive finite reference dimensions in cm")
    input_dir = args.input_dir.resolve()
    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file())
    if not paths:
        parser.error(f"No jpg/jpeg/png images in {input_dir}")
    if len({p.stem for p in paths}) != len(paths):
        parser.error("Duplicate image stems would overwrite masks; rename the inputs")
    output = ROOT / "output/recognition" / args.test_name
    if output.exists():
        parser.error(f"Output already exists: {output}. Choose a new test-name to preserve prior results.")
    manifest_by_path = {}
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        manifest_by_path = {(args.manifest.resolve().parent / item["file"]).resolve(): item for item in manifest["images"]}
        if set(paths) != set(manifest_by_path):
            parser.error("Manifest files must exactly match the input directory")
    hashes = {}
    for path in paths:
        hashes[path] = file_sha256(path)
        if args.manifest and hashes[path] != manifest_by_path[path]["sha256"]:
            raise ValueError(f"Input SHA256 mismatch: {path.name}")

    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from detectron2.config import get_cfg
    from detectron2.engine.defaults import DefaultPredictor
    from detectron2.projects.point_rend import add_pointrend_config
    from estimation.object_detection import stopsign_config, stopsign_meta, waterdepth_by_stopsign
    from test_image_seg import predict_one

    torch.set_num_threads(2)
    image_weights = ROOT / "records/link_efficientb4_model.pth"
    detector_weights = ROOT / "records/detectron2/model_final_ba17b9.pkl"
    if not image_weights.is_file() or not detector_weights.is_file():
        raise FileNotFoundError("Run bash .codex/setup.sh first; official model weights are missing")
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(stopsign_config["config_file"])
    cfg.MODEL.WEIGHTS = str(detector_weights)
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.freeze()
    detector = DefaultPredictor(cfg)
    # Official SHA-verified checkpoint contains a serialized model object.
    image_model = torch.load(image_weights, map_location="cpu").eval()
    for name in ("mask", "overlay", "detections", "depth"):
        (output / name).mkdir(parents=True, exist_ok=True)
    if dimensions[0] is not None:
        stopsign_meta.update(size=dimensions[0], pole_height=dimensions[1])
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    report = {
        "schema": "vfloodnet_depth_screening_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": commit, "test_name": args.test_name, "device": "cpu", "torch": torch.__version__,
        "reference_dimensions_cm": {"sign_size": dimensions[0], "pole_height": dimensions[1]},
        "demo_assumptions": args.demo, "accuracy_validated": False,
        "interpretation": "Model predictions only; null depth means unavailable, never zero depth. Candidate event links do not certify time/location. Frames from one source are not independent observations.",
        "summary": {}, "images": [],
    }
    for index, path in enumerate(paths, 1):
        print(f"[{index}/{len(paths)}] {path.name}", flush=True)
        predict_one(str(path), image_model, str(output / "mask"), str(output / "overlay"), torch.device("cpu"))
        mask = np.array(Image.open(output / "mask" / f"{path.stem}.png"))
        img = cv2.imread(str(path))
        with torch.inference_mode():
            instances = detector(img)["instances"].to("cpu")
        detections = []
        canvas = img.copy()
        counts = {"person": 0, "vehicle": 0, "stop_sign": 0}
        names = {0: ("person", "person"), 2: ("car", "vehicle"), 3: ("motorcycle", "vehicle"),
                 5: ("bus", "vehicle"), 7: ("truck", "vehicle"), 11: ("stop sign", "stop_sign")}
        for cls, score, box in zip(instances.pred_classes.tolist(), instances.scores.tolist(), instances.pred_boxes.tensor.tolist()):
            if cls not in names:
                continue
            label, key = names[cls]
            counts[key] += 1
            detections.append({"class": label, "confidence": score, "bbox_xyxy": box})
            x1, y1, x2, y2 = [round(x) for x in box]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 190, 255), 2)
            cv2.putText(canvas, f"{label} {score:.2f}", (max(0, x1), max(14, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.imwrite(str(output / "detections" / f"{path.stem}.png"), canvas)
        depth = None
        if not np.any(mask):
            status = "no_water_pixels"
        elif not counts["stop_sign"]:
            status = "no_stop_sign"
        elif dimensions[0] is None:
            status = "reference_dimensions_missing"
        else:
            _, candidate = waterdepth_by_stopsign(img, instances, mask, str(output / "depth"), path.stem)
            if math.isfinite(float(candidate)) and 0 <= candidate <= dimensions[1]:
                depth = float(candidate)
                status = "demo_assumptions" if args.demo else "estimated_unvalidated"
            else:
                status = "reference_geometry_failed"
        source = manifest_by_path.get(path, {})
        row = {
            "image_id": path.stem, "filename": path.name, "sha256": hashes[path],
            "model_water_fraction": float(np.count_nonzero(mask) / mask.size),
            "reference_counts": counts, "detections": detections,
            "depth_status": status, "estimated_depth_cm": depth,
            "ground_truth_depth_m": None, "accuracy_validated": False,
        }
        for key in ("source_urls", "related_event_ids", "observed_time", "location", "scene_type", "preprocessing_required", "reference_dimensions_verified", "review_state"):
            if key in source:
                row[key] = source[key]
        report["images"].append(row)
        report["summary"] = {
            "processed": len(report["images"]), "expected": len(paths),
            "numeric_estimates": sum(r["estimated_depth_cm"] is not None for r in report["images"]),
            "status_counts": dict(Counter(r["depth_status"] for r in report["images"])),
            "images_with_person": sum(r["reference_counts"]["person"] > 0 for r in report["images"]),
            "images_with_vehicle": sum(r["reference_counts"]["vehicle"] > 0 for r in report["images"]),
            "images_with_stop_sign": sum(r["reference_counts"]["stop_sign"] > 0 for r in report["images"]),
        }
        write_results(output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Results: {output / 'index.html'}")


if __name__ == "__main__":
    main()
