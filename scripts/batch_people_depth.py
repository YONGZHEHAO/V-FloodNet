"""Run visually approved body candidates, sharing the public network and SMPL."""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import time

if __package__:
    from .people_depth import ROOT, SMPL_NAME, build_network, infer_person, load_body_geometry
else:
    from people_depth import ROOT, SMPL_NAME, build_network, infer_person, load_body_geometry


def reviewed_targets(data):
    selected = data["selected"]
    ids = [row["person_id"] for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate person IDs in selected candidates")
    approved = [row for row in selected if row.get("reviewed_for_mesh") is True]
    if not approved:
        raise ValueError("No visually reviewed candidates: set reviewed_for_mesh=true after inspecting the target")
    if any(row.get("height_cm") is not None for row in approved):
        raise ValueError("This batch entry leaves heights unknown; use the single-person entry with a verified height instead")
    return approved


def save_batch_report(output, report):
    (output / "batch_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = ["person_id", "status", "submerged_body_fraction", "metric_depth_cm", "inference_seconds", "review_decision", "review_notes", "review_note", "error"]
    with (output / "results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for result in report["results"]:
            review = result.get("post_inference_review", {})
            writer.writerow({**result, "review_decision": review.get("decision"), "review_notes": review.get("notes")})
    cards = []
    for result in report["results"]:
        name = html.escape(result["person_id"])
        fraction = result.get("submerged_body_fraction")
        value = "不可估计" if fraction is None else f"{fraction:.2%}"
        reason = html.escape(result.get("review_note", "需检查网格与水线"))
        diagnostic = result.get("evidence", {}).get("diagnostic_image")
        image = f'<a href="{html.escape(diagnostic)}"><img loading="lazy" src="{html.escape(diagnostic)}" alt="目标人物、人体顶点、水体及模板水线"></a>' if diagnostic else ""
        post_review = result.get("post_inference_review", {})
        notes = html.escape(post_review.get("notes", "尚未完成运行后的目视复核；不能据此认为水深有效。"))
        priority = "true" if post_review.get("priority_for_relative_review") is True else "false"
        quality = "优先复核 · 尚未验证" if priority == "true" else "需要修正或进一步复核"
        cards.append(f'<article id="{name}" data-priority="{priority}"><h2>模型原始比例：{value} · {quality}</h2><p class="small">{name}</p><p>{reason}</p><p class="note">{notes}</p>{image}<p>厘米水深：未计算（真实身高未知） · <a href="{name}.json">原始记录</a> · 状态：{html.escape(result["status"])}</p></article>')
    skipped = "".join(f'<li>{html.escape(row["person_id"])}：{html.escape(row.get("pre_inference_review", {}).get("reason", "未获准用于网格推理"))}</li>' for row in report["skipped"])
    contents = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>人物估深批量结果</title>
<style>body{margin:0;background:#f3f5f7;color:#17242c;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1250px;margin:auto;padding:24px}article,.intro{background:white;padding:22px;border-radius:12px;margin:18px 0}h1{font-size:28px}h2{font-size:21px}img{width:100%;height:auto}.note{background:#fff3d6;padding:12px;border-left:4px solid #a66b00}.small,li{font-size:13px;color:#566}a{color:#075f96}button{font:inherit;padding:8px 14px;cursor:pointer;margin:4px;border:1px solid #adbfc8;border-radius:6px;background:white}</style><main><h1>人物估深批量结果</h1>'''
    execution = html.escape(report.get("execution_location", "Linux 推理环境"))
    contents += f'<div class="intro"><p>{execution} · 选择 {report["candidate_count"]} 个候选，批准试跑 {report["approved_count"]} 个，完成 {report["completed_count"]} 个。</p>'
    contents += '<p class="note"><strong>缺少已知人物身高，厘米水深均保留为空。</strong>原始比例未经现场真值验证；对位、水线、雨衣、靴子和多人遮挡仍需逐图复核。模型输出数值并不代表有效测量。</p><p>绿色框是明确选定的人物；蓝点是掩膜判水，橙点是非水；红线只表示模板中的估计水线。</p><p><a href="batch_results.json">完整结果</a> · <a href="results.csv">结果表 CSV</a></p></div>'
    summary = report.get("review_summary", {})
    if summary:
        contents += f'<p>运行后目视复核：{summary.get("priority_for_relative_review", 0)} 例对位较好，可优先复核；其余存在明显误差或较高不确定性。对位较好也不代表比例准确。</p>'
    contents += '<p><button type="button" onclick="showPeople(\'all\')">全部试跑</button><button type="button" onclick="showPeople(\'priority\')">优先复核</button><button type="button" onclick="showPeople(\'other\')">需修正／复核</button></p>'
    contents += "\n".join(cards) + f'<details><summary>跳过的 {len(report["skipped"])} 个候选及原因</summary><ul>{skipped}</ul></details>'
    contents += '''<script>function showPeople(mode){document.querySelectorAll('article[data-priority]').forEach(el=>{el.hidden=mode==='priority'?el.dataset.priority!=='true':mode==='other'?el.dataset.priority==='true':false;});}</script></main></html>'''
    (output / "index.html").write_text(contents, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smpl-path", type=Path, default=Path(os.environ.get("VFLOOD_SMPL_PATH", ROOT / "records/smpl" / SMPL_NAME)))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new output directory to preserve previous results")
    if not args.smpl_path.is_file():
        parser.error("Official licensed SMPL file is missing; import your own file first")
    data = json.loads(args.candidates.read_text(encoding="utf-8"))
    try:
        approved = reviewed_targets(data)
    except ValueError as error:
        parser.error(str(error))
    for row in approved:
        for key in ("crop_image", "crop_mask"):
            if not (args.candidates.parent / row[key]).is_file():
                parser.error(f"Missing {key} for {row['person_id']}")
    import torch
    torch.set_num_threads(2)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True)
    report = {"device": args.device, "candidate_count": len(data["selected"]), "approved_count": len(approved),
              "completed_count": 0, "error_count": 0, "batch_complete": False, "accuracy_validated": False,
              "candidates_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
              "code_sha256": {name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
                              for name in ("people_depth.py", "batch_people_depth.py")},
              "skipped": [r for r in data["selected"] if r.get("reviewed_for_mesh") is not True], "results": []}
    save_batch_report(args.output_dir, report)
    start = time.perf_counter()
    network = build_network(device)
    smpl, sampler = load_body_geometry(args.smpl_path, device)
    report["model_initialization_seconds"] = time.perf_counter() - start
    print(json.dumps({"model_loaded": True, "approved": len(approved), "device": args.device}), flush=True)
    for index, row in enumerate(approved, 1):
        started = time.perf_counter()
        try:
            result = infer_person(network, smpl, sampler, row, args.candidates.parent, device, args.output_dir)
            report["completed_count"] += 1
        except Exception as error:
            result = {"person_id": row["person_id"], "status": "inference_failed", "error": f"{type(error).__name__}: {error}",
                      "metric_depth_cm": None, "submerged_body_fraction": None}
            report["error_count"] += 1
        result["inference_seconds"] = time.perf_counter() - started
        result["review_note"] = row.get("pre_inference_review", {}).get("reason", "Visually approved for a mesh trial, not validated depth")
        report["results"].append(result)
        save_batch_report(args.output_dir, report)
        print(json.dumps({"progress": f"{index}/{len(approved)}", "person_id": row["person_id"], "status": result["status"],
                          "fraction": result["submerged_body_fraction"], "seconds": result["inference_seconds"]}), flush=True)
    report["batch_complete"] = True
    report["total_seconds"] = time.perf_counter() - start
    save_batch_report(args.output_dir, report)
    if report["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
