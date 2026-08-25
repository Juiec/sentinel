# COCO Hard-Negative Images for Bag Detection

Curated set of COCO images that contain **confusable objects** (person / chair / keyboard)
but contain **no real bag object**, so a bag detector trained on them learns to reject
people, chairs and keyboards instead of flagging them as bags.

## Why these are good negatives
Your dataset is 2952 positive vs 7 negative in train (~421:1). A model trained with that
ratio will happily box people/chairs/keyboards as "bags". These images give it clean
examples where the confusable object is present and correctly labeled *nothing*.

## How they were chosen (deterministic, seed=42)
From COCO train2017 (IDs already in your `coco_negative_ids.json`), keep images that:
  - contain at least one of {person, chair, keyboard} with a reasonably large bbox
    (area >= 4000 px²), AND
  - contain NO backpack / handbag / suitcase object.

Then balanced by confusable type:
  - 60 with keyboard (the most bag-shaped object)
  - 40 with person+chair
  - 30 chair-only
  - 20 person-only
= 150 images total, all from COCO train2017 (won't collide with your val split).

## Files
- coco_negative_selection.json   manifest: id, file_name, url, confusables present
- download_coco_negatives.py     resumable downloader (skip existing files)
- images/                        downloaded JPEGs land here

## Usage
    cd negatives_coco
    python download_coco_negatives.py           # all 150
    python download_coco_negatives.py --limit 20  # quick test

## Wiring into your YOLO dataset (merged_bags)
Negatives need EMPTY label files. Copy images + create empty labels:

    for f in $(ls images/*.jpg | sed 's|images/||'); do
        cp images/$f ../merged_bags/train/images/
        touch ../merged_bags/train/labels/${f%.jpg}.txt
    done

(Or better: put them in a `negatives` split and add it to your train list.)

## Notes
- COCO license (CC BY 4.0). Attribution required if you publish the model/dataset.
- If you later want more negatives, bump the group quotas in the selection script —
  ~49k clean candidates are available.
