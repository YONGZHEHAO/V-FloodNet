"""Build contact sheets and a review page from saved vehicle-level predictions."""
import argparse
import csv
import hashlib
import html
import json
from collections import Counter
from pathlib import Path
from urllib.parse import unquote
import re


def overlap_conflicts(detections):
    conflicts = []
    for index, a in enumerate(detections):
        for b in detections[index+1:]:
            if a["level"] == b["level"]:
                continue
            x, y = a["bbox_xyxy"], b["bbox_xyxy"]
            intersection = max(0, min(x[2], y[2])-max(x[0], y[0]))*max(0, min(x[3], y[3])-max(x[1], y[1]))
            area_a = max(0, x[2]-x[0])*max(0, x[3]-x[1])
            area_b = max(0, y[2]-y[0])*max(0, y[3]-y[1])
            union = area_a+area_b-intersection
            iou = intersection/union if union else 0
            containment = intersection/min(area_a, area_b) if min(area_a, area_b) else 0
            if iou >= .3 or containment >= .8:
                conflicts.append({"detection_ids": [a["detection_id"], b["detection_id"]],
                                  "levels": [a["level"], b["level"]], "iou": iou,
                                  "intersection_over_smaller_box": containment,
                                  "interpretation": "Possible duplicate or overlapping vehicles; manual review required, no automatic merge."})
    return conflicts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    output = args.results_dir
    report = json.loads((output / "results.json").read_text(encoding="utf-8"))
    reviews = json.loads(args.reviews.read_text(encoding="utf-8")) if args.reviews else {"images": []}
    review_map = {r["image_id"]: r for r in reviews["images"]}
    image_ids = {r["image_id"] for r in report["images"]}
    if not set(review_map).issubset(image_ids):
        raise ValueError("Manual review refers to an unknown image")
    from PIL import Image, ImageDraw
    sheets = []
    for start in range(0, len(report["images"]), 16):
        sheet = Image.new("RGB", (1440, 1120), "white")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(report["images"][start:start+16]):
            with Image.open(output / row["annotated_image"]) as source:
                image = source.copy()
            image.thumbnail((350, 230))
            x, y = (i%4)*360, (i//4)*280
            sheet.paste(image, (x+(360-image.width)//2, y))
            draw.text((x+3, y+233), f'#{start+i+1} {row["image_id"][:29]}', fill="black")
            draw.text((x+3, y+249), 'Levels: '+','.join(str(d["level"]) for d in row["detections"])[:40], fill="black")
            draw.text((x+3, y+264), row["status"], fill="black")
        filename = f'contact_{start//16+1:02d}.jpg'
        sheet.save(output / filename, quality=94)
        sheets.append(filename)
    conflicts = {r["image_id"]: c for r in report["images"] if (c := overlap_conflicts(r["detections"]))}
    flags = {"method": "Geometric screening only; may also flag overlapping distinct cars.",
             "conflict_image_count": len(conflicts), "images": conflicts}
    (output / "overlap_flags.json").write_text(json.dumps(flags, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    source_hash_count = 0
    for row in report["images"]:
        source = args.input_dir / row["filename"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != row["source_sha256"]:
            raise ValueError(f'Source hash mismatch: {row["image_id"]}')
        source_hash_count += 1
        assert row["estimated_depth_cm"] is None and not row["accuracy_validated"]
        for detection in row["detections"]:
            assert detection["estimated_depth_cm"] is None and not detection["accuracy_validated"]
    links_checked = 0
    for target in re.findall(r'(?:href|src)="([^"]+)"', (output / "index.html").read_text(encoding="utf-8")):
        if target.startswith(("https:", "http:", "#")):
            continue
        if not (output / unquote(target)).exists():
            raise ValueError(f"Broken HTML link: {target}")
        links_checked += 1
    summary = {"images_processed": len(report["images"]), "source_sha256_checked": source_hash_count,
               "raw_gallery_local_links_checked": links_checked, "reviewed_images": len(review_map),
               "visual_review_categories": dict(Counter(r["review_category"] for r in reviews["images"])),
               "overlapping_different_level_images": len(conflicts), "metric_depths": 0,
               "accuracy_validated": False, "raw_results_modified": False}
    (output / "verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    with (output / "visual_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "model_status", "visual_review_category", "reason", "estimated_depth_cm", "accuracy_validated"])
        for row in report["images"]:
            manual = review_map.get(row["image_id"], {})
            writer.writerow([row["image_id"], row["status"], manual.get("review_category", "not_individually_reviewed"), manual.get("reason", ""), "", False])
    cards = []
    category_labels = {"overlapping_labels": "同一车辆出现不同等级框", "partial_wheel_evidence": "可见局部车轮，仅作候选",
                       "wet_road_and_mixed_labels": "湿路面与远处混合等级", "false_vehicle": "误把其他物体当作车辆",
                       "occlusion_and_overlapping_labels": "遮挡与重叠异级框", "visible_inundation_candidate": "可见淹没关系，等级待验证",
                       "suspected_underestimate": "等级疑似低估", "waterline_occluded": "真实水线被遮挡",
                       "wet_road_unverifiable_levels": "反光路面，等级缺少可见依据"}
    # Show reviewed examples first, preserving the raw detector page separately.
    ordered = sorted(report["images"], key=lambda r: (r["image_id"] not in review_map, r["image_id"]))
    for row in ordered:
        manual = review_map.get(row["image_id"])
        note = html.escape(manual["reason"]) if manual else "本次未进行完整原图逐目标复核，保留为未验证候选。"
        category = html.escape(category_labels.get(manual["review_category"], manual["review_category"])) if manual else "未逐目标复核"
        conflict = "存在重叠异级框，请核查重复检测或遮挡。" if row["image_id"] in conflicts else ""
        levels = ", ".join(f'L{d["level"]} ({d["confidence"]:.2f})' for d in row["detections"]) or "无检测"
        cards.append(f'<article><h2>{html.escape(row["image_id"])}</h2><a href="{row["annotated_image"]}"><img loading="lazy" src="{row["annotated_image"]}"></a><p>模型候选：{levels}</p><p><b>视觉复核：{category}</b> · {note}</p><p class="warning">{conflict}</p></article>')
    overview = ''.join(f'<a href="{s}">概览{i+1}</a> · ' for i, s in enumerate(sheets))
    findings = ("本批复核已发现误检或疑似等级误判。" if any(r["review_category"] in {"false_vehicle", "suspected_underestimate"} for r in reviews["images"])
                else "本批尚无真实水深精度验证，请逐目标检查模型候选。")
    warning = findings + "输出仅适合辅助筛选，不能自动形成可信的水深数据集。没有检测不代表没有水，等级不换算厘米。"
    style = 'body{font:16px/1.7 system-ui;background:#f3f5f8;color:#172536;max-width:1150px;margin:auto;padding:24px}header,article{background:white;padding:22px;margin:18px 0;border-radius:12px}h2{font-size:15px;overflow-wrap:anywhere}img{width:100%;max-height:700px;object-fit:contain}.warning{color:#a33b22}pre{white-space:pre-wrap}'
    (output / "review.html").write_text(f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>车辆等级结果与复核</title><style>{style}</style><header><h1>车辆淹没等级：结果与复核</h1><p class="warning">{warning}</p><p><a href="index.html">原始模型结果</a> · <a href="results.csv">图片表</a> · <a href="detections.csv">目标表</a> · <a href="overlap_flags.json">重叠框筛查</a> · <a href="verification.json">文件检查</a></p><pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre><p>{overview}</p></header>'+''.join(cards)+'</html>', encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
