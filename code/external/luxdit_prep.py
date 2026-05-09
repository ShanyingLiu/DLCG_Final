"""Stage the test-split sphere PNGs into a flat directory for LuxDiT inference.

Files are named {eval_pos:05d}.png so we can map back to the dataloader order
when post-processing. Also writes eval_index.json:
  {"<eval_pos>": <metadata.json index>, ...}
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from config import config
from dataset.data import SphereDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--copy_metadata", action="store_true",
                    help="also dump metadata.json subset for the test split")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Build test split exactly like the evaluator does — note we use a no-op
    # transform here since we want the original PNG, not a tensor.
    ds = SphereDataset("test", config, transform=T.ToTensor())
    n = len(ds)
    print(f"[luxdit-prep] test split size: {n}")

    eval_index = {}
    for eval_pos in range(n):
        real_idx = int(ds.indices[eval_pos])
        src = ds.image_paths[real_idx]
        dst = os.path.join(args.out_dir, f"{eval_pos:05d}.png")
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        eval_index[str(eval_pos)] = real_idx

    with open(os.path.join(args.out_dir, "eval_index.json"), "w") as f:
        json.dump(eval_index, f, indent=2)

    if args.copy_metadata:
        with open(os.path.join(config.metadata_root, "metadata.json")) as f:
            full = json.load(f)
        sub = [full[ds.indices[i]] for i in range(n)]
        with open(os.path.join(args.out_dir, "metadata_test.json"), "w") as f:
            json.dump(sub, f, indent=2)

    print(f"[luxdit-prep] staged {n} files into {args.out_dir}")
    print(f"[luxdit-prep] index → {os.path.join(args.out_dir, 'eval_index.json')}")


if __name__ == "__main__":
    main()
