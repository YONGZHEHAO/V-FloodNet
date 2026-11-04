"""Fit reviewed wheel references and retain a status for every vehicle image.

No new detector or assumed wheel dimensions are used. Missing dimensions
always keep metric water depth null; rim fractions must not be called tire
fractions or compared directly with human-body fractions.
"""
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil

if __package__:
    from .wheel_geometry import estimate_annotation
else:
    from wheel_geometry import estimate_annotation

LEVELS = {"reference_partial_submergence": "车轮参考物局部淹没，几何试算",
          "partial_wheel_submergence": "可见车轮局部淹没",
          "water_reaches_vehicle_body": "可见水至车身下部或更高",
          "visible_vehicle_water_contact": "可见车辆接触积水，具体等级不清",
          "unavailable": "水线或参照物无法可靠判读"}
REFERENCES = {"rim": "轮毂", "outer_tire": "整条轮胎"}


def draw_diagnostic(image, annotation, result, path):
    import numpy as np
    from PIL import Image, ImageDraw
    ellipse = result["ellipse"]
    center = np.array(ellipse["center_xy"])
    theta = np.deg2rad(ellipse["angle_degrees"])
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    angles = np.linspace(0, 2*np.pi, 361)
    curve = np.column_stack([np.cos(angles), np.sin(angles)]) * ellipse["radii"]
    curve = curve @ rotation.T + center
    xmin, ymin = curve.min(axis=0)-22
    xmax, ymax = curve.max(axis=0)+22
    box = (max(0, int(xmin)), max(0, int(ymin)), min(image.width, int(xmax)+1), min(image.height, int(ymax)+1))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Fitted reference does not intersect the source image")
    original_crop = image.crop(box)
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    draw.line([tuple(p) for p in curve], fill=(0, 230, 255), width=1)
    for x, y in annotation["boundary_points_xy"]:
        draw.ellipse((x-1, y-1, x+1, y+1), fill=(255, 140, 0))
    first, second = annotation["waterline_xy"]
    slope = (second[1]-first[1])/(second[0]-first[0])
    line = [(x, first[1]+slope*(x-first[0])) for x in (box[0], box[2])]
    for shift in (-annotation["waterline_uncertainty_px"], annotation["waterline_uncertainty_px"]):
        draw.line([(x, y+shift) for x, y in line], fill=(255, 150, 150), width=1)
    draw.line(line, fill=(255, 0, 0), width=1)
    draw.ellipse((center[0]-1, center[1]-1, center[0]+1, center[1]+1), fill=(0, 230, 255))
    annotated_crop = overlay.crop(box)
    panels = []
    for crop in (original_crop, annotated_crop):
        target = crop.resize((520, max(1, round(520*crop.height/crop.width))), Image.Resampling.NEAREST)
        panels.append(target)
    height = max(p.height for p in panels)
    canvas = Image.new("RGB", (1040, height+90), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "Original reference crop", fill="black")
    draw.text((528, 8), "Manual landmarks + fitted ellipse + waterline", fill="black")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index*520, 30))
    draw.text((8, height+42), f'REFERENCE = {annotation["reference_kind"]}; fraction = {result["reference_submergence_fraction"]:.3f}; not metric depth', fill="black")
    draw.text((8, height+60), "Orange: visible boundary points. Cyan: fitted ellipse. Red: waterline. Pale red: line uncertainty.", fill="black")
    canvas.save(path)


def save_report(output, report):
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    fields = ["index", "image_id", "decision", "visual_level", "reference_kind", "reference_submergence_fraction",
              "sensitivity_low", "sensitivity_high", "metric_depth_cm", "reason"]
    with (output / "results.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in report["images"]:
            estimates = row.get("geometry_results", [])
            for estimate in estimates or [{}]:
                interval = estimate.get("annotation_sensitivity_p05_p95", [None, None])
                writer.writerow({**row, **estimate, "sensitivity_low": interval[0], "sensitivity_high": interval[1]})
    with (output / "coverage.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_id", "vehicle_detections", "route", "vehicle_review_decision"])
        writer.writeheader()
        decisions = {r["image_id"]: r["decision"] for r in report["images"]}
        writer.writerows({**r, "vehicle_review_decision": decisions.get(r["image_id"], "not_in_this_vehicle_batch")} for r in report["coverage"])
    cards = []
    for row in report["images"]:
        title = LEVELS[row["visual_level"]]
        detail = ""
        for estimate in row.get("geometry_results", []):
            if estimate.get("reference_submergence_fraction") is not None:
                lo, hi = estimate["annotation_sensitivity_p05_p95"]
                label = REFERENCES[estimate["reference_kind"]]
                detail += f'<p><strong>{label}淹没高度比例：{estimate["reference_submergence_fraction"]:.1%}</strong> · 标注扰动敏感性范围 {lo:.1%}–{hi:.1%}（不是准确率区间）</p><a href="{html.escape(estimate["diagnostic_image"])}"><img loading="lazy" src="{html.escape(estimate["diagnostic_image"])}" alt="参考物边缘人工点、水线及拟合"></a>'
                detail += f'<p>比例分母是完整{label}的直径，不是面积、人物身高或地面水深。当前结果尚未经实测验证。</p>'
            else:
                detail += f'<p class="note">几何试算失败：{html.escape(estimate.get("error", "unknown"))}</p>'
        cards.append(f'<article data-kind="{row["decision"]}"><h2>#{row["index"]} · {html.escape(title)}</h2><p class="small">{html.escape(row["image_id"])}</p><p>{html.escape(row["target_description"])}：{html.escape(row["reason"])}</p>{detail}<a href="{row["preview"]}"><img class="source" loading="lazy" src="{row["preview"]}" alt="原照片预览"></a><p>厘米水深：未计算；物理尺寸未核实。</p></article>')
    summary = report["summary"]
    contents = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>车辆参考物估深复核</title><style>body{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;background:#f3f5f7;color:#17242c;margin:0}main{max-width:1200px;margin:auto;padding:24px}article,.intro{background:white;border-radius:12px;padding:22px;margin:18px 0}h1{font-size:28px}h2{font-size:21px}img{max-width:100%;height:auto}.source{max-height:550px;object-fit:contain}.note{padding:12px;background:#fff3d6;border-left:4px solid #a66b00}.small{font-size:13px;color:#566}button{font:inherit;padding:8px 14px;cursor:pointer;margin:4px;border:1px solid #adbfc8;border-radius:6px;background:white}a{color:#075f96}</style><main><h1>车辆参考物估深复核</h1>'''
    vehicle_count = sum(len(r["vehicles"]) for r in report["images"])
    contents += f'<div class="intro"><p>原始照片 {report["total_input_images"]} 张；本轮检查 {len(report["images"])} 张车辆候选图、{vehicle_count}个原检测框。</p><p>几何试算成功 {summary["successful_geometry_references"]} 个参考物；仅定性判断 {summary["decision_counts"].get("qualitative_only",0)} 张；其余保留具体不可估原因。</p>'
    contents += '<p class="note">本轮采用人工标记的可见轮毂边缘和水线，程序自动拟合。没有训练新车轮检测器。轮毂比例不能直接当作轮胎比例或人物淹没比例；所有厘米水深保持为空。标注扰动范围不是现场误差或统计准确率保证。</p>'
    contents += '<p><a href="results.csv">车辆结果表</a> · <a href="coverage.csv">全部照片去向</a> · <a href="results.json">完整结果</a> · <a href="annotations.json">人工标注原始坐标</a> · <a href="README.md">复现说明</a></p></div>'
    contents += '<p><button onclick="filterImages(\'all\')">全部候选图</button><button onclick="filterImages(\'geometry_trial\')">几何试算</button><button onclick="filterImages(\'qualitative_only\')">定性等级</button><button onclick="filterImages(\'unavailable\')">不可估原因</button></p>'
    contents += "\n".join(cards)
    contents += '''<script>function filterImages(mode){document.querySelectorAll('article[data-kind]').forEach(el=>{el.hidden=mode!=='all'&&el.dataset.kind!==mode;});}</script></main></html>'''
    (output / "index.html").write_text(contents, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vehicles", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new output directory to preserve previous estimates")
    candidates = json.loads(args.vehicles.read_text(encoding="utf-8"))
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    reviews = {r["image_id"]: r for r in annotations["image_reviews"]}
    if len(reviews) != len(annotations["image_reviews"]) or set(reviews) != {r["image_id"] for r in candidates["images"]}:
        parser.error("Provide exactly one review decision for every candidate image")
    if len({a["wheel_id"] for a in annotations["wheel_annotations"]}) != len(annotations["wheel_annotations"]):
        parser.error("Wheel IDs must be unique")
    annotated_images = {a["image_id"] for a in annotations["wheel_annotations"]}
    if annotated_images != {r["image_id"] for r in reviews.values() if r["decision"] == "geometry_trial"}:
        parser.error("Every geometry-trial image must have at least one explicit reference annotation")
    for annotation in annotations["wheel_annotations"]:
        if annotation.get("reviewed") is not True or annotation["reference_kind"] not in {"rim", "outer_tire"}:
            parser.error("Only explicitly reviewed rim/outer-tire annotations are allowed")
        if annotation["image_id"] not in reviews or reviews[annotation["image_id"]]["decision"] != "geometry_trial":
            parser.error("A wheel annotation must belong to an image approved for geometry")
    from PIL import Image
    output = args.output_dir
    (output / "images").mkdir(parents=True)
    (output / "diagnostics").mkdir()
    shutil.copyfile(args.annotations, output / "annotations.json")
    rows, failed = [], 0
    for candidate in candidates["images"]:
        row = {**candidate, **reviews[candidate["image_id"]], "geometry_results": []}
        # Keep source detection evidence without links to crops outside this result package.
        row["vehicles"] = [{key: value for key, value in v.items() if key != "crop_image"} for v in row["vehicles"]]
        if row["decision"] not in {"geometry_trial", "qualitative_only", "unavailable"} or row["visual_level"] not in LEVELS:
            raise ValueError("Unknown review decision or visual level")
        path = args.input_dir / row["filename"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["source_sha256"]:
            raise ValueError(f"Source hash mismatch: {row['image_id']}")
        with Image.open(path) as source:
            image = source.convert("RGB")
        image.save(output / row["preview"], quality=92)
        for annotation in annotations["wheel_annotations"]:
            if annotation["image_id"] != row["image_id"]:
                continue
            if annotation["source_sha256"] != row["source_sha256"]:
                raise ValueError("Annotation and candidate identify different source bytes")
            for points in (annotation["boundary_points_xy"], annotation["waterline_xy"]):
                if any(not (0 <= x < image.width and 0 <= y < image.height) for x, y in points):
                    raise ValueError("Annotation coordinates fall outside the original image")
            try:
                estimate = {**estimate_annotation(annotation), "wheel_id": annotation["wheel_id"], "reference_kind": annotation["reference_kind"]}
                diagnostic = f'diagnostics/{annotation["wheel_id"]}.png'
                draw_diagnostic(image, annotation, estimate, output / diagnostic)
                estimate["diagnostic_image"] = diagnostic
            except ValueError as error:
                estimate = {"wheel_id": annotation["wheel_id"], "reference_kind": annotation["reference_kind"],
                            "status": "geometry_unavailable", "error": str(error), "reference_submergence_fraction": None,
                            "metric_depth_cm": None, "accuracy_validated": False}
                failed += 1
            row["geometry_results"].append(estimate)
            print(json.dumps(estimate), flush=True)
        rows.append(row)
    summary = {"decision_counts": dict(Counter(r["decision"] for r in rows)),
               "successful_geometry_references": sum(e["reference_submergence_fraction"] is not None for r in rows for e in r["geometry_results"]),
               "failed_geometry_references": failed, "metric_depth_count": 0, "accuracy_validated_count": 0}
    report = {"schema": "vfloodnet_wheel_review_results_v1", "total_input_images": candidates["total_input_images"],
              "input_vehicles_sha256": hashlib.sha256(args.vehicles.read_bytes()).hexdigest(),
              "annotations_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
              "code_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                              for name in ("estimate_wheel_depth.py", "wheel_geometry.py")},
              "method": "Manual landmark review + ellipse geometry under a local affine circle projection; not a new detector",
              "reference_kind_warning": "A rim fraction is not a tire fraction, a body fraction or ground water depth.",
              "summary": summary, "images": rows, "coverage": candidates["coverage"]}
    save_report(output, report)
    print(json.dumps(summary), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
