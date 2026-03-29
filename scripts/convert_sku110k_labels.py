"""Convert SKU-110K labels from YOLO center format (cx cy w h) to polygon corner format.

Our data loader's load_valid_labels() treats label values as polygon coordinate pairs
and computes bboxes via min/max. SKU-110K uses standard YOLO center format which would
produce incorrect bounding boxes. This script converts to 4-corner polygon format:
  class_id x1 y1 x2 y1 x2 y2 x1 y2
"""

import sys
from pathlib import Path


def convert_center_to_corners(line: str) -> str:
    parts = line.strip().split()
    if len(parts) != 5:
        return line  # skip malformed lines
    cls, cx, cy, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    # 4-corner polygon: top-left, top-right, bottom-right, bottom-left
    return f"{cls} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}\n"


def main():
    labels_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../yolo-finetune/datasets/SKU-110K/labels")

    txt_files = list(labels_dir.glob("*.txt"))
    print(f"Converting {len(txt_files)} label files in {labels_dir}")

    for label_file in txt_files:
        lines = label_file.read_text().strip().split("\n")
        converted = [convert_center_to_corners(line) for line in lines if line.strip()]
        label_file.write_text("".join(converted))

    # Spot check
    sample = txt_files[0] if txt_files else None
    if sample:
        print(f"\nSample ({sample.name}):")
        print(sample.read_text()[:500])

    print(f"\nDone. Converted {len(txt_files)} files.")


if __name__ == "__main__":
    main()
