#!/usr/bin/env python3
"""Build ``list.txt`` (one ``<image> <labelme_json>`` per line) from folders.

    python build_list.py --images data/png --jsons data/json --out data/list.txt

Images and LabelMe files are matched by stem (``0001.png`` <-> ``0001.json``).
Both DICOM-derived PNG/JPG and the LabelMe ``imagePath`` field are tolerated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--jsons", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-json", action="store_true",
                    help="verify each JSON parses and has >=1 polygon")
    args = ap.parse_args()

    img_dir, json_dir = Path(args.images), Path(args.jsons)
    jsons = {p.stem: p for p in json_dir.rglob("*.json")}
    lines, missing_json, missing_img = [], [], []
    for img in sorted(img_dir.rglob("*")):
        if img.suffix.lower() not in IMG_EXT:
            continue
        j = jsons.get(img.stem)
        if j is None:
            missing_json.append(img.name)
            continue
        if args.check_json:
            try:
                d = json.loads(j.read_text(encoding="utf-8"))
                if not any(len(s.get("points") or []) >= 3 for s in d.get("shapes", [])):
                    print(f"  warn: no polygon in {j.name}")
            except Exception as exc:
                print(f"  skip unparsable {j.name}: {exc}")
                continue
        lines.append(f"{img.as_posix()} {j.as_posix()}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}: {len(lines)} pairs")
    if missing_json:
        print(f"  {len(missing_json)} image(s) without JSON, e.g. {missing_json[:3]}")
    if missing_img:
        print(f"  {len(missing_img)} JSON(s) without image, e.g. {missing_img[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
