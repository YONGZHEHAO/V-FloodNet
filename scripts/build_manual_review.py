"""Package original photos and immutable model suggestions into an offline review app."""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--recognition-results", type=Path)
    parser.add_argument("--people-results", type=Path)
    parser.add_argument("--wheel-results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    if output.exists():
        parser.error("Use a new output directory to preserve previous work")
    report_path = args.results_dir / "results.json"
    report = read(report_path)
    prior = {r["image_id"]: r for r in read(args.recognition_results)["images"]} if args.recognition_results else {}
    visual_path = args.results_dir / "visual_review.json"
    visual = {r["image_id"]: r for r in read(visual_path)["images"]} if visual_path.exists() else {}
    overlap_path = args.results_dir / "overlap_flags.json"
    overlaps = read(overlap_path)["images"] if overlap_path.exists() else {}
    wheel = {r["image_id"]: r for r in read(args.wheel_results)["images"]} if args.wheel_results else {}
    people = read(args.people_results).get("results", []) if args.people_results else []
    for row in report["images"]:
        if Path(row["filename"]).name != row["filename"]:
            raise ValueError("Image filename must be flat")
        if digest(args.input_dir / row["filename"]) != row["source_sha256"]:
            raise ValueError(f'Source hash mismatch: {row["image_id"]}')
    output.mkdir(parents=True)
    (output / "originals").mkdir()
    (output / "evidence").mkdir()
    rows, copied_evidence = [], set()
    def evidence(source, name):
        if not source.is_file():
            raise FileNotFoundError(source)
        if name not in copied_evidence:
            shutil.copyfile(source, output / "evidence" / name)
            copied_evidence.add(name)
        return f"evidence/{name}"
    for ordinal, row in enumerate(report["images"], 1):
        image_id = row["image_id"]
        old = prior.get(image_id, {})
        advisor = visual.get(image_id, {})
        group = "false_positive" if advisor.get("review_category") == "false_vehicle" else (
            "conflict" if image_id in overlaps or advisor.get("review_category") == "suspected_underestimate"
            else "candidates" if row["detections"] else "no_detection")
        extra = []
        for person in people:
            if person["person_id"].rsplit("__p", 1)[0] != image_id:
                continue
            review = person.get("post_inference_review", {})
            name = person["evidence"]["diagnostic_image"]
            extra.append({"kind": "person", "target_id": person["person_id"],
                          "image": evidence(args.people_results.parent / name, name),
                          "fraction": person.get("submerged_body_fraction"),
                          "note": review.get("notes", "尚未验证的人物投影"),
                          "priority_for_relative_review": review.get("priority_for_relative_review", False)})
        wheel_row = wheel.get(image_id, {})
        for estimate in wheel_row.get("geometry_results", []):
            path = Path(estimate["diagnostic_image"])
            name = "wheel_" + path.name
            extra.append({"kind": "rim", "target_id": estimate["wheel_id"],
                          "image": evidence(args.wheel_results.parent / path, name),
                          "fraction": estimate.get("reference_submergence_fraction"),
                          "note": "人工轮毂几何试算，对标注敏感；不是轮胎比例或厘米水深。"})
        shutil.copyfile(args.input_dir / row["filename"], output / "originals" / row["filename"])
        rows.append({"ordinal": ordinal, "image_id": image_id, "filename": row["filename"],
                     "source_sha256": row["source_sha256"], "original": "originals/"+row["filename"],
                     "width": row["width"], "height": row["height"], "group": group,
                     "source_urls": [u for u in row.get("source_urls", []) if u.startswith("https://")],
                     "location": old.get("location", ""), "observed_time": old.get("observed_time", ""),
                     "prior_reference_counts": old.get("reference_counts", {}),
                     "model_status": row["status"], "detections": row["detections"],
                     "advisory_note": advisor.get("reason", ""), "overlap_flags": overlaps.get(image_id, []),
                     "prior_wheel_note": wheel_row.get("reason", ""), "evidence": extra})
    data = {"schema": "flood_manual_review_dataset_v1", "dataset_id": digest(report_path),
            "title": "118张积水照片 · 人工复核", "model_sha256": report["model_sha256"],
            "model_source": report["source_repository"], "image_count": len(rows),
            "detection_count": sum(len(r["detections"]) for r in rows),
            "group_counts": dict(Counter(r["group"] for r in rows)), "images": rows}
    js = "window.REVIEW_DATA = " + json.dumps(data, ensure_ascii=False).replace("<", "\\u003c") + ";\n"
    (output / "data.js").write_text(js, encoding="utf-8")
    assets = Path(__file__).resolve().parent / "manual_review_app"
    for name in ("index.html", "app.css", "app.js"):
        shutil.copyfile(assets / name, output / name)
    (output / "使用说明.txt").write_text(
        "积水照片人工复核\n\n1. 解压后双击 index.html；无需安装软件或联网。\n"
        "2. 左侧优先看“误检提示”和“等级冲突”，再看车辆候选和未检出图片。\n"
        "3. 先判是否可见积水、是否有可用参照物，再逐个框确认目标与等级。\n"
        "4. 我的提示只供参考，所有人工结论初始均为待复核。\n"
        "5. 未检出照片也可以点击“补框”后拖出人物或其他参照物；新框不会改写模型结果。\n"
        "6. 快捷键：J下一张，K上一张，B显隐检测框，Enter完成并下一张；输入文字时不触发。\n"
        "7. 页面会在当前浏览器暂存；请经常点“导出备份JSON”。换浏览器、移动文件夹或清理缓存前先备份。\n"
        "8. 下次用“导入JSON”恢复；只接受同批照片及同一数据版本的记录。导入前请先导出已有记录。\n"
        "9. “导出复核CSV”适合Excel查看，但CSV不能代替JSON恢复工作。\n"
        "10. 完成全部后，把导出的JSON给Codex，便可继续整理有效样本。\n\n"
        "注意：L0至L4是模型类别，不自动映射厘米。水线相对位置为人工视觉判断，不是实测水深。\n"
        "人物身高比例与轮毂直径比例不能混用；已有试算图仅作参照，缺尺寸时厘米深度为空。\n"
        "浏览器中的保存只在本机发生，页面不会上传图片或结论。\n", encoding="utf-8")
    checks = {"source_hashes_checked": len(rows), "copied_originals": len(rows),
              "copied_diagnostic_images": len(copied_evidence), "group_counts": data["group_counts"],
              "initial_user_reviews": 0, "inference_rerun": False, "dataset_id": data["dataset_id"]}
    (output / "package_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == "__main__":
    main()
