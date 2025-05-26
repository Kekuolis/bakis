# eval_checkpoints.py

import os
import glob
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse

from downstream import MultiNoisyFileDataset
from aten_nuate import VariantATENNuate

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate all aTENNuate checkpoints on a validation set (by MSE loss), with resume support"
    )
    p.add_argument("--clean_dir",   default="./irasai/",  help="Path to clean .wav files")
    p.add_argument("--noisy_dir",   default="./irasai/NOISY/",  help="Path to matching noisy .wav files")
    p.add_argument("--factor",      type=int, default=8,    help="Noisy variants per clean")
    p.add_argument("--sample_rate", type=int, default=16000,help="Resample rate")
    p.add_argument("--target_len",  type=int, default=16000,help="Window length in samples")
    p.add_argument("--batch_size",  type=int, default=8,    help="Validation batch size")
    p.add_argument("--ckpt_dir",    default="./checkpoints",help="Directory with *.pth files")
    p.add_argument("--out_json",    default="./eval_results.json",
                   help="Where to write the aggregated metrics")
    return p.parse_args()

def load_existing_results(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def numeric_sort(paths):
    return sorted(
        paths,
        key=lambda p: int(
            os.path.basename(p)
              .rsplit("_e", 1)[1]
              .split(".pth")[0]
        )
    )

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # build val loader
    full_ds = MultiNoisyFileDataset(
        clean_dir   = args.clean_dir,
        noisy_dir   = args.noisy_dir,
        factor      = args.factor,
        sample_rate = args.sample_rate,
        target_len  = args.target_len,
        debug_crop  = False
    )
    val_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=False)

    # gather checkpoints
    ckpt_paths = glob.glob(os.path.join(args.ckpt_dir, "*.pth"))
    variants = {}
    for path in ckpt_paths:
        name   = os.path.basename(path)
        prefix = name.rsplit("_e", 1)[0]
        variants.setdefault(prefix, []).append(path)

    # load (or init) results
    all_results = load_existing_results(args.out_json)
    criterion   = nn.MSELoss()

    # evaluate
    for prefix, paths in variants.items():
        # ensure list exists
        existing = { r["epoch"] for r in all_results.get(prefix, []) }
        all_results.setdefault(prefix, [])

        # sort by epoch number
        for ckpt_path in numeric_sort(paths):
            # extract epoch
            epoch = int(
                os.path.basename(ckpt_path)
                  .rsplit("_e",1)[1]
                  .split(".pth")[0]
            )
            if epoch in existing:
                print(f"[{prefix}] skipping epoch {epoch}, already done")
                continue

            ckpt = torch.load(ckpt_path, map_location=device)
            v    = ckpt.get("variant", {})
            model = VariantATENNuate(**v).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            # compute avg val loss
            total_loss, n = 0.0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    total_loss += criterion(pred, y).item()
                    n += 1
            avg_loss = total_loss / n

            print(f"[{prefix}] epoch {epoch:>2} → val_loss = {avg_loss:.4f}")
            all_results[prefix].append({
                "epoch":    epoch,
                "val_loss": avg_loss
            })

            # write JSON after each checkpoint
            os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
            with open(args.out_json, "w") as f:
                json.dump(all_results, f, indent=2)

    print(f"Done. Results written to {args.out_json}")

if __name__ == "__main__":
    main()
