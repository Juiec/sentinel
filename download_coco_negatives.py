#!/usr/bin/env python3
"""
Download curated COCO hard-negative images for bag-detection training.

These are images that contain confusable objects (person / chair / keyboard)
but NO real bag object, so the model learns "people/chairs/keyboards are not bags".

Usage:
    python download_coco_negatives.py            # downloads everything missing
    python download_coco_negatives.py --limit 20 # quick test run
"""
import json, os, sys, urllib.request

MANIFEST = "coco_negative_selection.json"
OUT_DIR  = "images"          # final images land here (negatives_coco/images/)
BASE_URL = "http://images.cocodataset.org/"

def main():
    limit = None
    args = sys.argv[1:]
    if "--limit" in args:
        i = args.index("--limit")
        limit = int(args[i+1])
    if "-h" in args or "--help" in args:
        print(__doc__); return

    manifest = json.load(open(MANIFEST))
    os.makedirs(OUT_DIR, exist_ok=True)
    items = manifest[:limit] if limit else manifest

    todo = [m for m in items if not os.path.exists(os.path.join(OUT_DIR, m["file_name"]))]
    print(f"{len(items)} selected, {len(todo)} to download (resume: skips existing)")
    ok = fail = 0
    for n, m in enumerate(items, 1):
        dest = os.path.join(OUT_DIR, m["file_name"])
        if not os.path.exists(dest):
            try:
                urllib.request.urlretrieve(m["url"], dest)
                ok += 1
            except Exception as e:
                print("FAIL", m["file_name"], e); fail += 1
        if n % 25 == 0 or n == len(items):
            print(f"  {n}/{len(items)}  (ok={ok} fail={fail})")

    print(f"done. ok={ok} fail={fail}")
    if fail:
        print("Re-run the same command to retry failures (resume-safe).")

if __name__ == "__main__":
    main()
