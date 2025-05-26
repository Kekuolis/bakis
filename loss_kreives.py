# eval_checkpoints.py

import os
import glob
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse

# imports from your codebase
from downstream import MultiNoisyFileDataset                          # as before
from aten_nuate import VariantATENNuate                               # as before

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate all aTENNuate checkpoints on a validation set (by MSE loss)"
    )
    p.add_argument("--clean_dir",    required=True, default="./irasai/",  help="Path to clean .wav files")
    p.add_argument("--noisy_dir",    required=True, default="./irasai/NOISY/",  help="Path to matching noisy .wav files")
    p.add_argument("--factor",       type=int, default=8,   help="Noisy variants per clean")
    p.add_argument("--sample_rate",  type=int, default=16000, help="Resample rate")
    p.add_argument("--target_len",   type=int, default=16000, help="Window length in samples")
    p.add_argument("--batch_size",   type=int, default=8,     help="Validation batch size")
    p.add_argument("--ckpt_dir",     default="./checkpoints",   help="Directory with *.pth files")
    p.add_argument("--out_json",     default="checkpoints/eval_results.json",
                   help="Where to write the aggregated metrics")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Build validation loader just like in main.py
    full_ds = MultiNoisyFileDataset(
        clean_dir   = args.clean_dir,
        noisy_dir   = args.noisy_dir,
        factor      = args.factor,
        sample_rate = args.sample_rate,
        target_len  = args.target_len,
        debug_crop  = False
    )
    val_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=False)

    # 2) Gather checkpoints
    ckpt_paths = sorted(glob.glob(os.path.join(args.ckpt_dir, "*.pth")))
    variants = {}
    for path in ckpt_paths:
        name = os.path.basename(path)
        prefix = name.rsplit("_e", 1)[0]
        variants.setdefault(prefix, []).append(path)

    # 3) We'll use MSELoss to evaluate
    criterion = nn.MSELoss()

    all_results = {}
    for prefix, paths in variants.items():
        all_results[prefix] = []
        for ckpt_path in sorted(paths):
            ckpt = torch.load(ckpt_path, map_location=device)
            # re-instantiate model
            v = ckpt.get("variant", {})
            model = VariantATENNuate(**v).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            # --- compute avg validation loss ---
            total_loss, n_batches = 0.0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    pred = model(x)
                    total_loss += criterion(pred, y).item()
                    n_batches += 1

            avg_loss = total_loss / n_batches
            epoch   = ckpt.get("epoch", None)
            print(f"[{prefix}] epoch {epoch:>2} → val_loss = {avg_loss:.4f}")

            all_results[prefix].append({
                "epoch":    epoch,
                "val_loss": avg_loss
            })

    # 4) Write out JSON
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved evaluation results to {args.out_json}")

if __name__ == "__main__":
    main()
