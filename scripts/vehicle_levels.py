"""Experimental FLOOD-DEPTH-ML CPU inference; classes are not measured depths."""
import argparse
import csv
import hashlib
import html
import importlib.metadata as metadata
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / ".codex/vendor/vehicle-levels"
WEIGHTS = ROOT / "records/vehicle_levels/FLOOD_DEPTH_ML_best_car.pt"
MODEL_SHA256 = "0b15b6114ba663d98509331faac467a1c838fb1f2c1235dfcf18fd63395b62ee"
SOURCE_COMMIT = "1c05473c537423b565051d6fec95dfdfeda05560"
MODEL_URL = f"https://raw.githubusercontent.com/mayankmi/FLOOD-DEPTH-ML/{SOURCE_COMMIT}/best_car.pt"
PACKAGES = {"ultralytics": "8.3.133", "ultralytics-thop": "2.0.18", "psutil": "5.9.8", "py-cpuinfo": "9.0.0", "seaborn": "0.12.2"}
COLORS = [(52, 132, 70), (34, 125, 178), (191, 115, 20), (202, 65, 65), (130, 48, 130)]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap():
    if not OVERLAY.is_dir():
        raise RuntimeError("Run bash .codex/setup_vehicle_levels.sh first")
    sys.path.insert(0, str(OVERLAY))
    # Keep generated settings and downloads on the project disk; disable telemetry/autoinstall.
    os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".codex/cache/ultralytics")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    from packaging.requirements import Requirement
    versions = {}
    for package, expected in PACKAGES.items():
        actual = metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"Expected {package}=={expected}, got {actual}")
        versions[package] = actual
        for raw in metadata.requires(package) or []:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            name = requirement.name
            # The locked base provides cv2 through the single contrib-headless wheel.
            if name == "opencv-python":
                name = "opencv-contrib-python-headless"
            installed = metadata.version(name)
            if requirement.specifier and installed not in requirement.specifier:
                raise RuntimeError(f"Unsatisfied {package} requirement: {raw}; got {installed}")
    import torch
    import cv2
    from ultralytics import YOLO, settings
    settings.update({"sync": False})
    versions.update(torch=torch.__version__, numpy=metadata.version("numpy"), opencv=cv2.__version__)
    return YOLO, versions


def download():
    if WEIGHTS.exists():
        if sha256(WEIGHTS) != MODEL_SHA256:
            raise ValueError("Existing vehicle checkpoint hash mismatch")
        print(json.dumps({"weights": str(WEIGHTS), "sha256": MODEL_SHA256, "cached": True}))
        return
    import requests
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    temporary = WEIGHTS.with_suffix(".download")
    with requests.get(MODEL_URL, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)
    if sha256(temporary) != MODEL_SHA256:
        raise ValueError("Downloaded vehicle checkpoint hash mismatch")
    temporary.replace(WEIGHTS)
    print(json.dumps({"weights": str(WEIGHTS), "sha256": MODEL_SHA256, "cached": False}))


def iou(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = max(0, a[2]-a[0])*max(0, a[3]-a[1]) + max(0, b[2]-b[0])*max(0, b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0


def render_gallery(output, report):
    cards = []
    for row in report["images"]:
        levels = ", ".join(f'L{d["level"]}: {d["confidence"]:.2f}' for d in row["detections"]) or "无检测"
        prior = html.escape(row.get("prior_wheel_reason") or "没有旧车辆复核记录")
        cards.append(f'<article data-status="{row["status"]}"><h2>{html.escape(row["image_id"])}</h2>'
                     f'<a href="{row["annotated_image"]}"><img loading="lazy" src="{row["annotated_image"]}"></a>'
                     f'<p><b>{html.escape(row["status"])}</b> · {html.escape(levels)}</p>'
                     f'<p>旧车辆检测：{row["prior_vehicle_count"]} · 旧车轮复核：{prior}</p>'
                     '<p>以上均为模型候选；未人工确认等级，厘米水深为空。</p></article>')
    style = 'body{font:16px/1.65 system-ui;background:#f2f5f8;max-width:1200px;margin:auto;padding:24px;color:#162230}article,header{background:white;padding:20px;margin:18px 0;border-radius:12px}h2{font-size:15px;overflow-wrap:anywhere}img{width:100%;max-height:700px;object-fit:contain}button{padding:9px 12px;margin:4px;border:1px solid #abc;border-radius:6px;cursor:pointer}.note{color:#90440e}pre{white-space:pre-wrap}'
    summary = html.escape(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    buttons = ''.join(f'<button onclick="filterCards(\'{s}\')">{s}</button>' for s in ["all", "matched_vehicle_candidate", "low_confidence_only", "unmatched_model_candidate", "no_detection"])
    script = "function filterCards(s){document.querySelectorAll('article').forEach(a=>a.hidden=s!=='all'&&a.dataset.status!==s)}"
    (output / "index.html").write_text(f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>车辆淹没等级试跑</title><style>{style}</style><header><h1>车辆淹没等级试跑</h1><p>FLOOD-DEPTH-ML · CPU · Level 0–4</p><p class="note">这是车辆目标的模型分级候选，不是照片整体水深。未检测到目标不表示零水深，置信度不等于准确率。本批没有真实水深标签，不能计算准确率。</p><p>matched_vehicle_candidate 仅表示置信度≥0.50且与旧汽车框IoU≥0.35；仍需人工确认。0.25–0.50仅作低置信候选。</p><p><a href="results.csv">图片表</a> · <a href="detections.csv">目标表</a> · <a href="results.json">完整记录</a></p><pre>{summary}</pre>{buttons}</header>'+''.join(cards)+f'<script>{script}</script></html>', encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--recognition-results", type=Path)
    parser.add_argument("--wheel-results", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--image-id", action="append", help="Select an image ID; repeat to select several.")
    parser.add_argument("--conf", type=float, default=.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    if args.download:
        download()
        return
    YOLO, versions = bootstrap()
    if args.check:
        print(json.dumps({"dependencies": versions, "overlay": str(OVERLAY), "checkpoint_loaded": False}))
        return
    if not all((args.input_dir, args.recognition_results, args.output_dir)):
        parser.error("--input-dir, --recognition-results and --output-dir are required")
    if not 0 < args.conf <= 1:
        parser.error("--conf must be between 0 and 1")
    if args.output_dir.exists():
        parser.error("Use a new output directory to preserve prior runs")
    if sha256(args.weights) != MODEL_SHA256:
        raise ValueError("Only the reviewed, fixed checkpoint is supported")
    source = json.loads(args.recognition_results.read_text(encoding="utf-8"))
    rows = source["images"]
    if args.image_id:
        selected = set(args.image_id)
        rows = [r for r in rows if r["image_id"] in selected]
        if {r["image_id"] for r in rows} != selected:
            raise ValueError("Unknown image ID")
    wheel = {}
    if args.wheel_results:
        wheel = {r["image_id"]: r for r in json.loads(args.wheel_results.read_text(encoding="utf-8"))["images"]}
    for row in rows:
        if sha256(args.input_dir / row["filename"]) != row["sha256"]:
            raise ValueError(f'Source image hash mismatch: {row["image_id"]}')
    started = time.perf_counter()
    model = YOLO(str(args.weights), task="detect")
    names = model.names
    if names != {i: f"Level {i}" for i in range(5)}:
        raise ValueError(f"Unexpected checkpoint classes: {names}")
    from PIL import Image, ImageDraw
    output = args.output_dir
    (output / "annotated").mkdir(parents=True)
    results = []
    for index, row in enumerate(rows, 1):
        path = args.input_dir / row["filename"]
        with Image.open(path) as original:
            image = original.convert("RGB")
        # Full-image detector, matching upstream; do not resize vehicle crops to fill a scene.
        prediction = model.predict(image, device="cpu", imgsz=args.imgsz, conf=args.conf, iou=.7,
                                   agnostic_nms=False, save=False, verbose=False)[0]
        draw = ImageDraw.Draw(image)
        detections = []
        prior_vehicles = [d for d in row["detections"] if d["class"] in {"car", "truck", "bus", "motorcycle"}]
        for box in prediction.boxes:
            coords = [float(v) for v in box.xyxy[0].tolist()]
            level = int(box.cls.item())
            confidence = float(box.conf.item())
            matches = sorted([(iou(coords, d["bbox_xyxy"]), d) for d in prior_vehicles], key=lambda t: t[0], reverse=True)
            best_iou, match = matches[0] if matches else (0., None)
            # Agreement on a car location is only a screening flag, never validation of the level.
            matched_car = best_iou >= .35 and match["class"] == "car"
            detection = {"detection_id": f'{row["image_id"]}__l{len(detections)+1:02d}', "level": level,
                         "label": names[level], "confidence": confidence, "bbox_xyxy": coords,
                         "matched_prior_vehicle_class": match["class"] if best_iou >= .35 else None,
                         "prior_vehicle_iou": best_iou, "matched_car_candidate": matched_car and confidence >= .5,
                         "manual_review": "pending", "estimated_depth_cm": None, "accuracy_validated": False}
            detections.append(detection)
            draw.rectangle(coords, outline=COLORS[level], width=max(2, round(image.width/500)))
            label = f'L{level} {confidence:.2f} #{len(detections)}'
            x, y = max(0, int(coords[0])), max(0, int(coords[1])-15)
            draw.rectangle((x, y, x+len(label)*6+5, y+15), fill=COLORS[level])
            draw.text((x+2, y+1), label, fill="white")
        annotated = f'annotated/{row["image_id"]}.jpg'
        image.save(output / annotated, quality=94)
        high = [d for d in detections if d["confidence"] >= .5]
        status = ("matched_vehicle_candidate" if any(d["matched_car_candidate"] for d in detections)
                  else "unmatched_model_candidate" if high
                  else "low_confidence_only" if detections else "no_detection")
        result = {"image_id": row["image_id"], "filename": row["filename"], "source_sha256": row["sha256"],
                  "source_urls": row.get("source_urls", []), "width": image.width, "height": image.height,
                  "prior_vehicle_count": len(prior_vehicles), "prior_wheel_decision": wheel.get(row["image_id"], {}).get("decision"),
                  "prior_wheel_reason": wheel.get(row["image_id"], {}).get("reason"), "status": status,
                  "annotated_image": annotated, "detections": detections,
                  "estimated_depth_cm": None, "accuracy_validated": False, "inference_speed_ms": prediction.speed}
        results.append(result)
        (output / "partial_results.json").write_text(json.dumps({"images": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"done": index, "total": len(rows), "image_id": row["image_id"], "levels": [d["level"] for d in detections], "status": status}), flush=True)
    summary = {"processed_images": len(results), "images_with_model_detections": sum(bool(r["detections"]) for r in results),
               "model_detection_count": sum(len(r["detections"]) for r in results),
               "status_counts": dict(Counter(r["status"] for r in results)),
               "level_detection_counts": dict(sorted(Counter(str(d["level"]) for r in results for d in r["detections"]).items())),
               "metric_depth_count": 0, "accuracy_validated_count": 0}
    report = {"schema": "vehicle_levels_experiment_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "method": "FLOOD-DEPTH-ML full-image YOLO11 vehicle inundation level detection",
              "source_repository": "https://github.com/mayankmi/FLOOD-DEPTH-ML", "source_commit": SOURCE_COMMIT,
              "model_sha256": MODEL_SHA256, "dependencies": versions,
              "parameters": {"device": "cpu", "imgsz": args.imgsz, "conf": args.conf, "iou": .7,
                             "candidate_confidence": .5, "prior_car_iou": .35},
              "source_recognition_sha256": sha256(args.recognition_results),
              "script_sha256": sha256(Path(__file__)), "elapsed_seconds": time.perf_counter()-started,
              "interpretation": "Unvalidated per-vehicle level candidates. No detection is not zero depth. No conversion to cm.",
              "summary": summary, "images": results}
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (output / "partial_results.json").unlink()
    with (output / "results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "status", "model_detection_count", "level_candidates", "prior_vehicle_count", "prior_wheel_decision", "estimated_depth_cm", "accuracy_validated"])
        for row in results:
            writer.writerow([row["image_id"], row["status"], len(row["detections"]), ";".join(f'L{d["level"]}:{d["confidence"]:.4f}' for d in row["detections"]), row["prior_vehicle_count"], row["prior_wheel_decision"], "", False])
    with (output / "detections.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["image_id", "detection_id", "level", "confidence", "matched_prior_vehicle_class", "prior_vehicle_iou", "matched_car_candidate", "manual_review", "estimated_depth_cm", "accuracy_validated"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            for detection in row["detections"]:
                writer.writerow({"image_id": row["image_id"], **detection})
    render_gallery(output, report)
    print(json.dumps({"summary": summary, "output": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
