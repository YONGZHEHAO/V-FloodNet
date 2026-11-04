"""Headless CPU-compatible METRO body inference, with explicit SMPL readiness.

Public checkpoint loading and an encoder CPU smoke check work without SMPL.
Real mesh inference requires the user's official licensed SMPL neutral file.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / ".codex/vendor/MeshTransformer"
SMPL_NAME = "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"


def configure_modules():
    sys.path.insert(0, str(VENDOR))
    from metro.modeling.bert import BertConfig, METRO, METRO_Body_Network
    from metro.modeling.hrnet.hrnet_cls_net_featmaps import get_cls_net
    from metro.modeling.hrnet.config import config, update_config
    return BertConfig, METRO, METRO_Body_Network, get_cls_net, config, update_config


def build_network(device):
    import torch
    BertConfig, METRO, Body, get_cls_net, hrnet_config, update_config = configure_modules()
    layers = []
    for input_dim, hidden_dim, output_dim in zip((2051, 512, 128), (1024, 256, 128), (512, 128, 3)):
        config = BertConfig.from_pretrained(str(VENDOR / "metro/modeling/bert/bert-base-uncased"))
        config.img_feature_dim, config.output_feature_dim = input_dim, output_dim
        config.hidden_size, config.num_hidden_layers, config.num_attention_heads = hidden_dim, 4, 4
        config.output_attentions = False
        config.output_hidden_states = False
        # Preserve the original checkpoint's legacy intermediate size (3072).
        layers.append(METRO(config))
    update_config(hrnet_config, str(ROOT / "records/people/hrnet_w64.yaml"))
    backbone = get_cls_net(hrnet_config, pretrained="")
    network = Body(SimpleNamespace(device=device), config, backbone, torch.nn.Sequential(*layers), None)
    state = torch.load(ROOT / "records/people/metro_3dpw_state_dict.bin", map_location="cpu")
    network.load_state_dict(state, strict=True)
    del state
    return network.to(device).eval()


def estimate_fraction(template, projected, mask):
    """Upstream template-boundary estimate, with empty/invalid cases retained."""
    import numpy as np
    if (template.ndim != 2 or template.shape[1] < 2 or projected.ndim != 2
            or projected.shape[1] != 2 or template.shape[0] != projected.shape[0]):
        raise ValueError("Expected aligned Nx3 (or Nx2) template and Nx2 projected mesh points")
    if not np.isfinite(template).all() or not np.isfinite(projected).all():
        return None
    resolution = mask.shape[0]
    points = np.clip(((projected + 1) * resolution / 2).astype(int), 0, resolution - 1)
    template_y = np.clip(((template[:, 1] + 1) * resolution / 2).astype(int), 0, resolution - 1)
    submerged = mask[points[:, 1], points[:, 0]] != 0
    if not submerged.any() or submerged.all():
        return None
    above_bottom = np.median(np.sort(template_y[~submerged])[-30:])
    below = template_y[submerged]
    below = below[below > above_bottom]
    if not below.size:
        return None
    boundary = (above_bottom + np.median(np.sort(below)[:10])) // 2
    height = np.ptp(template_y)
    if height <= 0:
        return None
    fraction = 1 - (boundary - template_y.min()) / height
    return float(fraction) if 0 <= fraction <= 1 else None


def save_diagnostics(image, mask, projected, template, fraction, prefix, target_box=None):
    """Save aligned evidence for reviewing a predicted body and its water labels."""
    import numpy as np
    from PIL import Image, ImageDraw

    size = mask.shape[0]
    pixels = np.clip(((projected + 1) * size / 2).astype(int), 0, size - 1)
    template_pixels = np.clip(((template[:, :2] + 1) * size / 2).astype(int), 0, size - 1)
    wet = mask[pixels[:, 1], pixels[:, 0]] != 0
    original = image.resize((size, size), Image.Resampling.BILINEAR)
    overlay_array = np.array(original).copy()
    water = mask != 0
    overlay_array[water] = (.6 * overlay_array[water] + .4 * np.array([0, 160, 255])).astype(np.uint8)
    overlay = Image.fromarray(overlay_array)
    projected_panel, template_panel = overlay.copy(), Image.new("RGB", (size, size), "white")
    for panel, points in ((projected_panel, pixels), (template_panel, template_pixels)):
        draw = ImageDraw.Draw(panel)
        for (x, y), is_wet in zip(points, wet):
            color = (0, 100, 255) if is_wet else (220, 90, 0)
            draw.ellipse((int(x)-1, int(y)-1, int(x)+1, int(y)+1), fill=color)
    if target_box is not None:
        for panel in (original, overlay, projected_panel):
            ImageDraw.Draw(panel).rectangle(target_box, outline=(0, 180, 70), width=1)
    if fraction is not None:
        boundary = template_pixels[:, 1].min() + (1 - fraction) * np.ptp(template_pixels[:, 1])
        ImageDraw.Draw(template_panel).line((0, int(round(boundary)), size-1, int(round(boundary))), fill="red", width=2)
    canvas = Image.new("RGB", (4 * size, size + 54), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (title, panel) in enumerate(zip(("Input crop", "Predicted water", "Projected body points", "Template water labels"),
                                          (original, overlay, projected_panel, template_panel))):
        draw.text((i * size + 4, 6), title, fill="black")
        canvas.paste(panel, (i * size, 24))
    draw.text((4, size + 33), "Green box: target. Blue: mask-water; orange: mask-dry. Red template line: estimated boundary. Unvalidated.", fill="black")
    diagnostic = prefix.with_name(prefix.name + "_diagnostic.png")
    canvas.resize((canvas.width * 2, canvas.height * 2), Image.Resampling.NEAREST).save(diagnostic)
    point_file = prefix.with_name(prefix.name + "_points.json")
    point_file.write_text(json.dumps({"coordinate_system": "normalized crop coordinates; -1 to +1 spans the crop",
                                      "projected_vertices_2d": projected.tolist(),
                                      "mask_water_at_clipped_vertices": wet.tolist()}, indent=2) + "\n")
    return {"diagnostic_image": diagnostic.name, "projected_points_file": point_file.name,
            "vertex_count": int(len(projected)), "mask_water_vertex_count": int(wet.sum()),
            "projected_vertices_outside_crop": int(((projected < -1) | (projected > 1)).any(axis=1).sum())}


def load_body_geometry(smpl_path, device):
    """Initialize the licensed body and sampler once for a single run or batch."""
    import metro.modeling.data.config as cfg
    cfg.SMPL_FILE = str(smpl_path.resolve())
    for name in ("JOINT_REGRESSOR_TRAIN_EXTRA", "JOINT_REGRESSOR_H36M_correct", "SMPL_sampling_matrix"):
        setattr(cfg, name, str(VENDOR / getattr(cfg, name)))
    from metro.modeling._smpl import SMPL, Mesh
    return SMPL().to(device), Mesh(filename=cfg.SMPL_sampling_matrix, device=device)


def infer_person(network, smpl, sampler, row, candidates_dir, device, output_dir, height_cm=None, report=None):
    """Run one explicit target with a shared network; keep missing height null."""
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from metro.utils.geometric_layers import orthographic_projection
    if height_cm is not None and not 0 < height_cm < 300:
        raise ValueError("Known height must be between 0 and 300 cm")
    image_path = candidates_dir / row["crop_image"]
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    with Image.open(candidates_dir / row["crop_mask"]) as source:
        mask = np.array(source.resize((224, 224), Image.Resampling.NEAREST))
    if image.width != image.height or mask.ndim != 2:
        raise ValueError("Expected a square image crop and an aligned label mask")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("Water mask must contain only background=0 and water=1 labels")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                    transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
    with torch.inference_mode():
        camera, _, vertices_sub2, _, _ = network(transform(image).unsqueeze(0).to(device), smpl, sampler)
        projected = orthographic_projection(vertices_sub2, camera).squeeze(0).cpu().numpy()
    if projected.shape != (431, 2) or not np.isfinite(projected).all():
        raise RuntimeError("Expected 431 finite projected body vertices")
    template = np.array(json.loads((ROOT / "records/template_3Dmesh.txt").read_text()))
    fraction = estimate_fraction(template, projected, mask)
    left, top, right, bottom = row["crop_xyxy"]
    target_box = [(row["bbox_xyxy"][i] - (left if i % 2 == 0 else top)) * 224 / (right - left)
                  for i in range(4)]
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = save_diagnostics(image, mask, projected, template, fraction, output_dir / row["person_id"], target_box)
    evidence["selected_person_bbox_224"] = target_box
    result = dict(report or {})
    result.update(person_id=row["person_id"], device=str(device), mesh_inference_run=True, submerged_body_fraction=fraction,
                  supplied_height_cm=height_cm,
                  metric_depth_cm=fraction * height_cm if fraction is not None and height_cm is not None else None,
                  source_sha256=row["source_sha256"], crop_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                  camera_parameters=camera.detach().cpu().reshape(-1).tolist(), evidence=evidence,
                  accuracy_validated=False, status="estimated_unvalidated" if fraction is not None else "water_boundary_unavailable")
    (output_dir / (row["person_id"] + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--load-weights", action="store_true", help="Strictly load the full public checkpoint without mesh inference.")
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--person-id", help="Explicit visually reviewed person ID from people.json.")
    parser.add_argument("--height-cm", type=float, help="Known height of this person; omitted keeps metric depth null.")
    parser.add_argument("--smpl-path", type=Path, default=Path(os.environ.get("VFLOOD_SMPL_PATH", ROOT / "records/smpl" / SMPL_NAME)))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/people_depth")
    args = parser.parse_args()
    if args.height_cm is not None and not 0 < args.height_cm < 300:
        parser.error("height-cm must be a known height between 0 and 300 cm")
    public_files = [ROOT / "records/people/metro_3dpw_state_dict.bin", ROOT / "records/people/hrnet_w64.yaml"]
    missing = [str(path) for path in public_files if not path.is_file()]
    smpl_ready = args.smpl_path.is_file()
    if not args.check and not smpl_ready:
        parser.error(f"Official SMPL file missing: {args.smpl_path}. Obtain it at https://smplify.is.tue.mpg.de/ .")
    report = {"device": args.device, "smpl_present": smpl_ready, "smpl_expected": str(args.smpl_path),
              "missing_public_files": missing, "full_checkpoint_loaded": False,
              "encoder_cpu_smoke": False, "mesh_inference_run": False, "metric_depth_cm": None,
              "ready_for_mesh_inference": False}
    import torch
    torch.set_num_threads(2)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable in this environment")
    BertConfig, METRO, *_ = configure_modules()
    config = BertConfig(vocab_size_or_config_json_file=100)
    config.hidden_size, config.num_attention_heads, config.num_hidden_layers = 24, 4, 1
    config.intermediate_size, config.img_feature_dim, config.output_feature_dim = 32, 7, 3
    model = METRO(config).eval()
    with torch.inference_mode():
        result = model(torch.zeros((1, 5, 7)))
    assert result.shape == (1, 5, 3) and torch.isfinite(result).all()
    report["encoder_cpu_smoke"] = True
    network = None
    if args.load_weights or not args.check:
        if missing:
            raise FileNotFoundError(f"Missing public files: {missing}; run bash .codex/setup_people.sh")
        network = build_network(torch.device(args.device))
        report["full_checkpoint_loaded"] = True
    report["ready_for_mesh_inference"] = smpl_ready and not missing
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.check:
        (args.output_dir / "readiness.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    if not smpl_ready:
        raise FileNotFoundError(f"Official SMPL file missing: {args.smpl_path}. Obtain it at https://smplify.is.tue.mpg.de/ .")
    if not args.candidates or not args.person_id:
        parser.error("Specify candidates and an explicit visually reviewed person-id")
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    selected = [row for row in candidates["selected"] if row["person_id"] == args.person_id]
    if len(selected) != 1:
        parser.error("person-id must identify one selected crop")
    row = selected[0]
    device = torch.device(args.device)
    smpl, sampler = load_body_geometry(args.smpl_path, device)
    report = infer_person(network, smpl, sampler, row, args.candidates.parent, device, args.output_dir, args.height_cm, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
