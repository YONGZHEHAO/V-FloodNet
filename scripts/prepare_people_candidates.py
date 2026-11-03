"""Rank detected people for visual review and export matched image/mask crops.

This is a screening aid, not a depth estimator or a pose/waterline validator.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def make_contact_sheet(output, selected):
    from PIL import Image, ImageDraw
    columns, cell = 5, 210
    rows = max(1, (len(selected) + columns - 1) // columns)
    canvas = Image.new("RGB", (columns * cell, rows * 240), "white")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(selected):
        with Image.open(output / row["crop_image"]) as source:
            thumb = source.convert("RGB")
            thumb.thumbnail((200, 200))
        x, y = (index % columns) * cell, (index // columns) * 240
        canvas.paste(thumb, (x + (cell - thumb.width) // 2, y))
        draw.text((x + 6, y + 203), f"#{index + 1}: " + row["image_id"][:20], fill="black")
        draw.text((x + 6, y + 220), row["person_id"].split("__")[-1], fill="black")
    canvas.save(output / "overview.jpg", quality=90)


def square_crop(box, width, height):
    x1, y1, x2, y2 = box
    size = min(int(1.5 * max(x2 - x1, y2 - y1)), width, height)
    left = max(0, min(width - size, round((x1 + x2 - size) / 2)))
    top = max(0, min(height - size, round((y1 + y2 - size) / 2)))
    return [left, top, left + size, top + size]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recognition-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    report = json.loads((args.recognition_dir / "results.json").read_text(encoding="utf-8"))
    if args.output_dir.exists():
        parser.error("Use a new output directory to preserve previous crops")
    if args.top <= 0:
        parser.error("top must be positive")
    rows = []
    for entry in report["images"]:
        people = [item for item in entry["detections"] if item["class"] == "person"]
        if not people:
            continue
        source_path = args.input_dir / entry["filename"]
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Input hash mismatch: {entry['filename']}")
        with Image.open(source_path) as image:
            width, height = image.size
        with Image.open(args.recognition_dir / "mask" / (entry["image_id"] + ".png")) as image:
            mask = np.array(image)
        if mask.shape != (height, width):
            raise ValueError(f"Mask alignment mismatch: {entry['filename']}")
        for number, person in enumerate(people, 1):
            x1, y1, x2, y2 = person["bbox_xyxy"]
            w, h = x2 - x1, y2 - y1
            left, right = max(0, int(x1 - .25 * w)), min(width, int(x2 + .25 * w))
            top, bottom = max(0, int(y1 + .65 * h)), min(height, int(y2 + .15 * h))
            region = mask[top:bottom, left:right]
            water_support = float(np.mean(region != 0)) if region.size else 0.0
            flags = []
            if h < 120 or w < 35:
                flags.append("small_person")
            if x1 < 2 or y1 < 2 or x2 > width - 2 or y2 > height - 2:
                flags.append("touches_image_edge")
            if person["confidence"] < .9:
                flags.append("low_detector_confidence")
            if water_support < .05:
                flags.append("little_model_water_near_lower_body")
            if w / max(h, 1) > .65:
                flags.append("wide_box_pose_needs_review")
            score = person["confidence"] * min(h / 350, 1) * min(water_support / .25, 1) / (1 + len(flags))
            rows.append({
                "person_id": f'{entry["image_id"]}__p{number:02d}',
                "image_id": entry["image_id"], "filename": entry["filename"],
                "source_sha256": entry["sha256"], "bbox_xyxy": person["bbox_xyxy"],
                "detector_confidence": person["confidence"], "bbox_height_px": h,
                "crop_xyxy": square_crop(person["bbox_xyxy"], width, height),
                "near_lower_body_model_water_fraction": water_support,
                "screening_flags": flags, "review_priority_score": score,
                "height_cm": None, "reviewed_for_mesh": False,
                "source_urls": entry.get("source_urls", []),
            })
    rows.sort(key=lambda item: item["review_priority_score"], reverse=True)
    selected = [row for row in rows if not row["screening_flags"]][:args.top]
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "crops").mkdir()
    cards = []
    for row in selected:
        stem = row["person_id"]
        with Image.open(args.input_dir / row["filename"]) as source:
            crop = source.convert("RGB").crop(row["crop_xyxy"])
            crop.save(args.output_dir / "crops" / f"{stem}.jpg", quality=95)
        with Image.open(args.recognition_dir / "mask" / (row["image_id"] + ".png")) as source:
            source.crop(row["crop_xyxy"]).save(args.output_dir / "crops" / f"{stem}_mask.png")
        row["crop_image"] = f"crops/{stem}.jpg"
        row["crop_mask"] = f"crops/{stem}_mask.png"
        cards.append(f'<article><h2>{html.escape(stem)}</h2><img src="{row["crop_image"]}">'
                     f'<p>检测置信度 {row["detector_confidence"]:.2f}；仍需人工检查姿态、遮挡、脚底和水线。</p></article>')
    make_contact_sheet(args.output_dir, selected)
    result = {"schema": "vfloodnet_person_candidates_v1", "recognition_commit": report["commit"],
              "screening_only": True, "detected_people": len(rows), "selected_for_visual_review": len(selected),
              "selection_rule": "confidence>=0.9, box height>=120px and width>=35px, no edge contact, width/height<=0.65, model-water support>=0.05 around lower body. These are heuristics, not proof of a valid waterline.",
              "selected": selected, "all_detections": rows}
    (args.output_dir / "people.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    page = ('<!doctype html><meta charset="utf-8"><title>人物候选复核</title><style>'
            'body{font:16px/1.6 system-ui;max-width:1100px;margin:30px auto;padding:20px}article{display:inline-block;vertical-align:top;width:45%;margin:2%;}h2{font-size:12px;overflow-wrap:anywhere}img{width:100%;max-height:500px;object-fit:contain}</style>'
            f'<h1>人物候选复核：{len(selected)}人</h1><p>从{len(rows)}个人物检测框中按尺寸、边界、检测分数和模型水体邻域筛选。'
            '裁剪与掩膜使用同一坐标；未运行人体网格模型，未确认水线，身高与厘米水深均未知。</p>' + ''.join(cards))
    (args.output_dir / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"detected_people": len(rows), "selected_for_visual_review": len(selected), "output": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
