"""Create a source-checked gallery for reviewing vehicle wheels and waterlines."""
import argparse
import hashlib
import html
import json
from pathlib import Path

VEHICLES = {"car", "truck", "bus", "motorcycle"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recognition-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--people-results", type=Path, help="Optionally exclude images already used in the body batch.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new output directory to preserve previous reviews")
    from PIL import Image, ImageDraw
    source_report = json.loads((args.recognition_dir / "results.json").read_text(encoding="utf-8"))
    excluded = set()
    if args.people_results:
        body = json.loads(args.people_results.read_text(encoding="utf-8"))
        excluded = {r["person_id"].rsplit("__p", 1)[0] for r in body["results"] if r.get("mesh_inference_run")}
    output = args.output_dir
    (output / "images").mkdir(parents=True)
    (output / "crops").mkdir()
    records, coverage = [], []
    for row in source_report["images"]:
        vehicles = [d for d in row["detections"] if d["class"] in VEHICLES]
        status = "prior_people_trial" if row["image_id"] in excluded else "vehicle_review" if vehicles else "no_vehicle_detection"
        coverage.append({"image_id": row["image_id"], "vehicle_detections": len(vehicles), "route": status})
        if status != "vehicle_review":
            continue
        path = args.input_dir / row["filename"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Source hash mismatch: {row['image_id']}")
        with Image.open(path) as original:
            image = original.convert("RGB")
        preview = f'images/{row["image_id"]}.jpg'
        image.save(output / preview, quality=94)
        candidates = []
        for index, vehicle in enumerate(vehicles, 1):
            x1, y1, x2, y2 = vehicle["bbox_xyxy"]
            pad = .15 * max(x2-x1, y2-y1)
            box = [max(0, int(x1-pad)), max(0, int(y1-pad)), min(image.width, int(x2+pad)), min(image.height, int(y2+pad))]
            vehicle_id = f'{row["image_id"]}__v{index:02d}'
            crop = f'crops/{vehicle_id}.jpg'
            image.crop(box).save(output / crop, quality=95)
            candidates.append({"vehicle_id": vehicle_id, **vehicle, "crop_xyxy": box, "crop_image": crop})
        records.append({"index": len(records)+1, "image_id": row["image_id"], "filename": row["filename"],
                        "source_sha256": row["sha256"], "width": image.width, "height": image.height,
                        "source_urls": row.get("source_urls", []), "preview": preview, "vehicles": candidates,
                        "model_water_fraction": row["model_water_fraction"], "wheel_review": None,
                        "metric_depth_cm": None})
    for start in range(0, len(records), 16):
        page = records[start:start+16]
        sheet = Image.new("RGB", (1440, 1120), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(page):
            with Image.open(output / row["preview"]) as source:
                thumb = source.copy()
            thumb.thumbnail((352, 240))
            x, y = (index % 4)*360, (index//4)*280
            sheet.paste(thumb, (x+(360-thumb.width)//2, y))
            draw.text((x+4, y+242), f'#{row["index"]}: {row["image_id"][:29]}', fill="black")
            draw.text((x+4, y+259), f'{row["width"]}x{row["height"]}; vehicles={len(row["vehicles"])}', fill="black")
        sheet.save(output / f'overview_{start//16+1}.jpg', quality=95)
    report = {"schema": "vfloodnet_vehicle_review_v1", "total_input_images": len(source_report["images"]),
              "excluded_prior_people_images": len(excluded), "vehicle_review_images": len(records),
              "interpretation": "Vehicle detection is not wheel detection or water-depth measurement. All metric depths remain null.",
              "images": records, "coverage": coverage}
    (output / "vehicles.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    cards = []
    for row in records:
        cards.append(f'<article><h2>#{row["index"]} · {html.escape(row["image_id"])}</h2><a href="{row["preview"]}"><img loading="lazy" src="{row["preview"]}"></a><p>{row["width"]}×{row["height"]} · 车辆检测{len(row["vehicles"])}个 · 尚未进行车轮与水线复核</p></article>')
    (output / "index.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>车辆与车轮复核</title><style>body{font:16px/1.6 system-ui;background:#f3f5f7;max-width:1200px;margin:auto;padding:24px}article{background:white;padding:20px;margin:20px 0}img{width:100%;max-height:700px;object-fit:contain}h2{font-size:16px}</style><h1>车辆与车轮复核</h1><p>检出车辆并不说明车轮可见或可用于估深。当前不填厘米水深。</p>'+'\n'.join(cards)+'</html>', encoding="utf-8")
    print(json.dumps({"images": len(records), "vehicles": sum(len(r["vehicles"]) for r in records), "output": str(output)}))


if __name__ == "__main__":
    main()
