"""Merge two or more YOLO-format datasets into one (plus optional negatives).

Each input dataset can be either of the two layouts:

    ds/                            (a) split-first layout  <- e.g. /bags
    |-- train/
    |   |-- images/                (.jpg/.png/.bmp)
    |   `-- labels/                (.txt, "class cx cy w h" normalized)
    |-- valid/
    |   |-- images/
    |   `-- labels/
    `-- data.yaml                  (optional; "names" used for class remapping)

    ds/                            (b) nested layout
    |-- images/
    |   |-- train/
    |   `-- val/
    |-- labels/
    |   |-- train/
    |   `-- val/
    `-- data.yaml                  (optional)

The merged output always uses the split-first layout:

    merged/
    |-- train/
    |   |-- images/
    |   `-- labels/
    |-- valid/
    |   ...
    `-- data.yaml

Class IDs are remapped by class NAME, so datasets with different class sets
merge without collisions. If a dataset has no data.yaml it is assumed to be
single-class ("object").

Negatives: images that contain NO objects (background / rejection samples).
They get an EMPTY .txt label, which YOLO trainers read as "no objects".
A negatives folder can be flat images, or a full dataset layout
(images/<split>/ + labels/<split>/, or <split>/images/ + <split>/labels/).

Usage:
    python merge_datasets.py /bags ds2 -o merged/
    python merge_datasets.py /bags --negatives neg/ -o merged/
    python merge_datasets.py /bags --negatives neg/ --neg-split train,valid -o merged/
"""

import argparse
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_names(ds_root):
    """Return {class_id: class_name} from a dataset's data.yaml (or default)."""
    yaml_path = ds_root / "data.yaml"
    if not yaml_path.exists():
        return {0: "object"}
    text = yaml_path.read_text(encoding="utf-8")

    # Prefer PyYAML when available
    try:
        import yaml
        data = yaml.safe_load(text)
        names = data.get("names", {})
        if isinstance(names, list):
            return {i: n for i, n in enumerate(names)}
        return {int(k): v for k, v in dict(names).items()}
    except ImportError:
        pass

    # Minimal fallback parser (no dependencies)
    names = {}
    lines = text.splitlines()
    idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("names:"):
            idx = i
            break
    if idx is None:
        return {0: "object"}

    head = lines[idx].split(":", 1)[1].strip()
    if head.startswith("{"):  # inline dict: names: {0: cat, 1: dog}
        for pair in head.strip("{}").split(","):
            pair = pair.strip()
            if not pair:
                continue
            k, v = pair.split(":", 1)
            names[int(k.strip())] = v.strip().strip('"').strip("'")
    else:
        for line in lines[idx + 1:]:
            s = line.strip()
            if not s or not line.startswith(" "):  # block ended
                break
            if s.startswith("-"):
                names[len(names)] = s.lstrip("- ").strip()
            else:
                k, v = s.split(":", 1)
                names[int(k.strip())] = v.strip().strip('"').strip("'")
    return names


def detect_layout(ds_root):
    """Return 'split_first' or 'nested' for a dataset root.

    split_first: <root>/<split>/images/ + <root>/<split>/labels/   (e.g. /bags/train/images)
    nested:      <root>/images/<split>/ + <root>/labels/<split>/
    """
    if (ds_root / "images").is_dir():
        return "nested"
    for sub in sorted(ds_root.iterdir()):
        if sub.is_dir() and (sub / "images").is_dir():
            return "split_first"
    sys.exit("No images folder found in {}.\nExpected <root>/<split>/images/ or <root>/images/<split>/.".format(ds_root))


def iter_splits(ds, layout):
    """Yield (split_name, images_dir, labels_dir) for a dataset root."""
    if layout == "nested":
        images_root = ds / "images"
        labels_root = ds / "labels"
        for split_dir in sorted(images_root.iterdir()):
            if not split_dir.is_dir():
                continue
            yield split_dir.name, split_dir, labels_root / split_dir.name
    else:  # split_first
        for split_dir in sorted(ds.iterdir()):
            if not split_dir.is_dir() or not (split_dir / "images").is_dir():
                continue
            yield split_dir.name, split_dir / "images", split_dir / "labels"


def build_remap(src_names, merged):
    """Map old class id -> new merged id, registering new names in `merged`."""
    remap = {}
    for old_id, name in src_names.items():
        if name not in merged:
            merged[name] = len(merged)
        remap[old_id] = merged[name]
    return remap


def rewrite_label(src, dst, remap):
    """Copy a label .txt, remapping class ids. Returns True on success."""
    lines = src.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        try:
            old_id = int(parts[0])
        except ValueError:
            return False  # malformed -> don't copy garbage
        new_id = remap.get(old_id)
        if new_id is None:
            print("  WARNING: label {} has unknown class id {}".format(src.name, old_id))
            continue
        out.append(" ".join([str(new_id)] + parts[1:]))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    return True


def collect_negative_sources(neg_root):
    """Yield (image_path, label_path_or_None) from a negatives folder.

    Supports:
      - flat images inside neg/images/ (labels in neg/labels/)   <- e.g. /negatives_coco
      - nested dataset layout (neg/images/<split>/ + neg/labels/<split>/)
      - split-first dataset layout (neg/<split>/images/ + neg/<split>/labels/)
      - flat images directly under neg/ (labels in neg/labels/ or next to the image)
    """
    if (neg_root / "images").is_dir():
        for entry in sorted((neg_root / "images").iterdir()):
            if entry.is_file():  # flat images inside images/
                if entry.suffix.lower() not in IMAGE_EXTS:
                    continue
                lbl = neg_root / "labels" / (entry.stem + ".txt")
                yield entry, (lbl if lbl.exists() else None)
            elif entry.is_dir():  # nested layout: images/<split>/...
                lbl_dir = neg_root / "labels" / entry.name
                for img in sorted(entry.iterdir()):
                    if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
                        continue
                    lbl = lbl_dir / (img.stem + ".txt")
                    yield img, (lbl if lbl.exists() else None)
    elif any(sub.is_dir() and (sub / "images").is_dir() for sub in neg_root.iterdir()):
        # split-first layout
        for split_dir in sorted(neg_root.iterdir()):
            imgs = split_dir / "images"
            if not split_dir.is_dir() or not imgs.is_dir():
                continue
            lbls = split_dir / "labels"
            for img in sorted(imgs.iterdir()):
                if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
                    continue
                lbl = (lbls / (img.stem + ".txt")) if lbls.is_dir() else None
                yield img, (lbl if lbl and lbl.exists() else None)
    else:  # flat images directly under neg root
        for img in sorted(neg_root.iterdir()):
            if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl = neg_root / "labels" / (img.stem + ".txt")
            if not lbl.exists():
                lbl = neg_root / (img.stem + ".txt")
                if not lbl.exists():
                    lbl = None
            yield img, lbl


def add_negatives(neg_roots, out_root, neg_splits):
    """Copy negative images into the merged dataset with EMPTY labels."""
    copied, warnings = 0, []
    for neg_root in neg_roots:
        if not neg_root.is_dir():
            sys.exit("Negatives folder not found: " + str(neg_root))

        for split in neg_splits:
            dst_imgs = out_root / split / "images"
            dst_labels = out_root / split / "labels"
            dst_imgs.mkdir(parents=True, exist_ok=True)
            dst_labels.mkdir(parents=True, exist_ok=True)

            for img, lbl in collect_negative_sources(neg_root):
                dest_img = dst_imgs / img.name
                if dest_img.exists():
                    warnings.append(img.name + ": already exists in " + split + ", skipped (not overwritten)")
                    continue
                if lbl is not None and lbl.read_text(encoding="utf-8").strip():
                    warnings.append(img.name + ": has a non-empty label file - not a true negative, skipped")
                    continue
                shutil.copy2(img, dest_img)
                (dst_labels / (img.stem + ".txt")).write_text("", encoding="utf-8")  # empty = no objects
                copied += 1
    return copied, warnings


def merge_datasets(ds_roots, out_root, neg_roots=None, neg_splits=("train",)):
    merged_names = {}          # name -> new id (insertion order)
    remaps = []                 # per-dataset old_id -> new_id
    layouts = []

    for ds in ds_roots:
        if not ds.is_dir():
            sys.exit("Dataset folder not found: " + str(ds))
        layout = detect_layout(ds)
        layouts.append(layout)
        names = load_names(ds)
        remaps.append(build_remap(names, merged_names))

    splits_present = set()
    copied_imgs, copied_labels, warnings = 0, 0, []

    for ds, layout, remap in zip(ds_roots, layouts, remaps):
        contributed = False
        for split, images_dir, labels_dir in iter_splits(ds, layout):
            if not images_dir.is_dir():
                sys.exit("No images folder found for split {} in {}".format(split, ds))
            contributed = True
            splits_present.add(split)
            dst_imgs = out_root / split / "images"
            dst_labels = out_root / split / "labels"
            dst_imgs.mkdir(parents=True, exist_ok=True)
            dst_labels.mkdir(parents=True, exist_ok=True)

            if not labels_dir.is_dir():
                warnings.append("{}/{}: no labels folder found".format(ds, split))

            for img in sorted(images_dir.iterdir()):
                if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
                    continue
                dest = dst_imgs / img.name
                if dest.exists():
                    warnings.append(img.name + ": already exists in " + split + ", skipped (not overwritten)")
                    continue
                shutil.copy2(img, dest)
                copied_imgs += 1

                label = labels_dir / (img.stem + ".txt")
                if label.exists():
                    rewrite_label(label, dst_labels / label.name, remap)
                    copied_labels += 1
                else:
                    warnings.append(img.name + ": no matching label found (copied image only)")

            # orphan labels (no matching image) are skipped
            if labels_dir.is_dir():
                for lbl in sorted(labels_dir.iterdir()):
                    if not any((dst_imgs / (lbl.stem + ext)).exists() for ext in IMAGE_EXTS):
                        warnings.append(lbl.name + ": no matching image found (label skipped)")
        if not contributed:
            warnings.append("{}: no splits with images/ found (nothing copied)".format(ds))

    # add negatives (empty labels) into the chosen split(s)
    neg_copied, neg_warnings = 0, []
    if neg_roots:
        neg_copied, neg_warnings = add_negatives(neg_roots, out_root, neg_splits)
        warnings.extend(neg_warnings)
        splits_present.update(neg_splits)

    # write merged data.yaml (negatives add NO new classes)
    lines = ["path: " + str(out_root.resolve())]
    for split in sorted(splits_present):
        lines.append(split + ": " + split + "/images")
    names_final = {}
    for name in merged_names:
        names_final[len(names_final)] = name
    lines.append("names:")
    for cid, cname in sorted(names_final.items()):
        lines.append("  " + str(cid) + ": " + cname)
    (out_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Merged {} dataset(s) into {}".format(len(ds_roots), out_root))
    print("  images copied: {}".format(copied_imgs + neg_copied))
    print("  labels copied: {}".format(copied_labels))
    print("  negatives added: {}".format(neg_copied))
    print("  classes:")
    for cid, cname in sorted(names_final.items()):
        print("    {}: {}".format(cid, cname))
    if warnings:
        print("  warnings:")
        for w in warnings:
            print("   - " + w)


def main():
    parser = argparse.ArgumentParser(description="Merge YOLO-format datasets (with optional negatives).")
    parser.add_argument("datasets", nargs="+",
                         help="Dataset folders. Each is either <root>/<split>/images/ + <root>/<split>/labels/"
                               " (e.g. /bags with train/, valid/, test/) or the nested <root>/images/<split>/ layout.")
    parser.add_argument("--negatives", action="append", default=None,
                         help="Folder of negative images (flat, nested, or split-first). Can be given multiple times.")
    parser.add_argument("--neg-split", default="train",
                         help="Comma-separated splits for negatives, e.g. 'train,valid' or 'both' "
                              "(shorthand for train,valid). Default: train")
    parser.add_argument("-o", "--output", default="merged", help="Output folder (default: merged)")
    args = parser.parse_args()

    ds_roots = [Path(d) for d in args.datasets]
    neg_roots = [Path(n) for n in args.negatives] if args.negatives else []
    neg_split_raw = args.neg_split.strip()
    if neg_split_raw.lower() == "both":
        neg_splits = ("train", "valid")
    else:
        neg_splits = tuple(s.strip() for s in neg_split_raw.split(",") if s.strip())

    out_root = Path(args.output)
    if len(ds_roots) < 1 and not neg_roots:
        sys.exit("Provide at least one dataset folder.")
    if out_root.exists() and any(out_root.iterdir()):
        sys.exit("Output folder {} already exists and is not empty. Delete it or use -o with a new name.".format(out_root))

    merge_datasets(ds_roots, out_root, neg_roots, neg_splits)


if __name__ == "__main__":
    main()
