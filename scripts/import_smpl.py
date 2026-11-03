"""Import only the required neutral SMPL file from the user's official archive.

No account cookies/passwords are read, and licensed bytes stay under records/.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--archive", type=Path)
    group.add_argument("--model", type=Path)
    args = parser.parse_args()
    target = ROOT / "records/smpl" / NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".pkl.part")
    try:
        if args.archive:
            with zipfile.ZipFile(args.archive) as archive:
                matches = [item for item in archive.infolist() if Path(item.filename).name == NAME and not item.is_dir()]
                if len(matches) != 1:
                    raise ValueError(f"Expected exactly one {NAME} in the official SMPLIFY_CODE_V2 archive; found {len(matches)}")
                member = matches[0]
                if not 1024 * 1024 < member.file_size < 256 * 1024 * 1024:
                    raise ValueError("Unexpected SMPL model size")
                with archive.open(member) as source, temporary.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
        else:
            if args.model.name != NAME:
                raise ValueError(f"Expected {NAME}")
            shutil.copyfile(args.model, temporary)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise FileExistsError("A different SMPL file is already installed; it has been preserved")
        else:
            temporary.replace(target)
        report = {"file": NAME, "sha256": digest, "bytes": target.stat().st_size,
                  "source": "User-supplied official SMPLify download; no publisher checksum was available",
                  "source_website": "https://smplify.is.tue.mpg.de/", "redistributed": False}
        output = ROOT / "output/people_depth"
        output.mkdir(parents=True, exist_ok=True)
        (output / "smpl-file-info.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
