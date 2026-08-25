"""Create empty label files for a folder of images.

For every image named like  <image_id>.jpg  (case-insensitive, also matches .jpeg)
this creates an empty file at  label/<image_id>.txt  so you can start writing labels.
Existing label files are left untouched (nothing is overwritten).

Usage:
    python make_label_files.py                  # images in ./images, labels in ./label
    python make_label_files.py my_photos out/   # custom folders
"""

import sys
from pathlib import Path


def main() -> None:
    images_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("images")
    label_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("label")

    if not images_dir.is_dir():
        sys.exit(f"Image folder not found: {images_dir}")

    label_dir.mkdir(parents=True, exist_ok=True)

    created, skipped = [], []
    for img in sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.jpeg")):
        txt_path = label_dir / (img.stem + ".txt")
        if txt_path.exists():
            skipped.append(txt_path.name)
        else:
            txt_path.touch()
            created.append(txt_path.name)

    print(f"Scanned {images_dir}: {len(created)} new label files, {len(skipped)} already existed.")
    for name in created:
        print("  created:", name)


if __name__ == "__main__":
    main()
