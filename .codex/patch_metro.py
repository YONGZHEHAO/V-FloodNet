"""Apply a small, checked CPU-device patch to an ignored pinned vendor checkout."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / ".codex/vendor/MeshTransformer"
SHA = "412b30142f39e4ac4480a54943993e232697c909"
actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=VENDOR, text=True).strip()
if actual != SHA:
    raise RuntimeError(f"Unexpected MeshTransformer commit: {actual}")
path = VENDOR / "metro/modeling/bert/modeling_metro.py"
source = path.read_text()
replacements = {
    "torch.zeros([batch_size, seq_length],dtype=torch.long).cuda()": "torch.zeros([batch_size, seq_length], dtype=torch.long, device=img_feats.device)",
    "template_pose.cuda(self.config.device)": "template_pose.to(images.device)",
    "torch.zeros((1,10)).cuda(self.config.device)": "torch.zeros((1,10), device=images.device)",
    "torch.ones_like(features).cuda(self.config.device)": "torch.ones_like(features)",
}
for before, after in replacements.items():
    if source.count(before) == 1:
        source = source.replace(before, after)
    elif source.count(after) != 1:
        raise RuntimeError(f"Cannot safely apply device patch: {before}")
path.write_text(source)
print("Pinned METRO body/encoder CPU device patch prepared; hand model is unchanged.")
