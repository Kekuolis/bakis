import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
import os
import glob
import math
from typing import List, Dict
import time
import re
from aten_nuate import VariantATENNuate

# --- Helper to find & load the latest checkpoint for one variant ----

def load_latest_checkpoint(model, optimizer, scheduler, prefix, ckpt_dir='checkpoints', device='cpu'):
    pattern = os.path.join(ckpt_dir, f"{prefix}_e*.pth")
    files   = glob.glob(pattern)
    if not files:
        print(f"⚠️  No checkpoints found for {prefix}")
        return 1

    def epoch_of(path):
        m = re.search(r"_e(\d+)\.pth$", path)
        return int(m.group(1)) if m else -1

    latest = max(files, key=epoch_of)
    ckpt   = torch.load(latest, map_location=device)

    model.load_state_dict(     ckpt['model_state_dict'])
    optimizer.load_state_dict( ckpt['optimizer_state_dict'])
    scheduler.load_state_dict( ckpt['scheduler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    print(f"🔄 Loaded {latest} (resuming at epoch {ckpt['epoch']})")
    return start_epoch

# --- Specify your variants exactly as in training --------------------

variants = [
    {'use_preconv': True,  'norm': 'layernorm', 'activation': 'silu'},
    {'use_preconv': False, 'norm': 'layernorm', 'activation': 'silu'},
    # {'use_preconv': True,  'norm': 'batchnorm', 'activation': 'relu'},
    # {'use_preconv': False, 'norm': 'batchnorm', 'activation': 'relu'},
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = {}

for v in variants:
    # a safe filesystem prefix
    prefix = f"preconv_{v['use_preconv']}_norm_{v['norm']}_act_{v['activation']}"
    print(f"\n=== Loading variant: {prefix} ===")

    # 1) instantiate model+opt+sch
    model     = VariantATENNuate(**v).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # 2) load latest checkpoint
    start_epoch = load_latest_checkpoint(
        model, optimizer, scheduler, prefix, ckpt_dir='checkpoints', device=device
    )

    # 3) switch to eval mode
    model.eval()

    # store for later
    models[prefix] = model

# Now `models` is a dict mapping each prefix to a ready‐to‐use model.
# Example usage:

# choose one
mymodel = models['preconv_False_norm_layernorm_act_silu']

# Path to your input and output files
# /home/kek/Documents/bakis/deep_state/irasai/test/NOISY/R_RD_F3_BG040_a0148_5db.wav
# /home/kek/Documents/bakis/deep_state/irasai/test/NOISY/R_RD_F3_BG040_a0148_40db.wav
in_path  = '/home/kek/Documents/bakis/deep_state/irasai/test/NOISY/R_RD_F3_BG040_a0148_5db.wav'
out_path = './output_enhanced.wav'

# Your model’s expected sample rate
MODEL_SR = 16000

# 1) Load and preprocess the .wav
waveform, sr = torchaudio.load(in_path)        # waveform: (channels, T)
# convert to mono if needed
if waveform.size(0) > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
# resample if needed
if sr != MODEL_SR:
    waveform = torchaudio.transforms.Resample(sr, MODEL_SR)(waveform)

# make it a batch of size 1 on the right device
waveform = waveform.unsqueeze(0).to(device)    # (1, 1, T)

# 2) Run the model
mymodel.eval()
with torch.no_grad():
    enhanced = mymodel(waveform)               # (1, 1, T)

# 3) Postprocess & save
enhanced = enhanced.squeeze(0).cpu()           # (1, T)
# optional: clamp to [-1,1]
enhanced = torch.clamp(enhanced, -1.0, 1.0)

# write out as WAV
torchaudio.save(out_path, enhanced, MODEL_SR)
print(f"Wrote enhanced file: {out_path}")
