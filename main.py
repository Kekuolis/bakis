import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
import os
import pprint
import glob
from downstream import *
from aten_nuate import VariantATENNuate
import math
from typing import List, Dict
import time


# 3) MAIN ENTRYPOINT --------------------------------------------------------

def load_waveforms_from_dir(
    directory: str,
    sample_rate: int = 16000
) -> List[torch.Tensor]:
    """
    Loads all .wav in `directory`, converts to mono, resamples to sample_rate,
    returns a list of (1, T) tensors.
    """
    wav_paths = sorted(glob.glob(os.path.join(directory, '*.wav')))
    waves = []
    resampler = None
    for p in wav_paths:
        wav, sr = torchaudio.load(p)            # (channels, T)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            if resampler is None or resampler.orig_freq != sr:
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
            wav = resampler(wav)
        waves.append(wav)                       # list of (1, T)
    return waves
def pad_collate_sr(batch):
    """
    batch: list of (y_sr, y) pairs, each shape (1, L_i)
    returns: (ysr_batch, y_batch) both (B, 1, L_max)
    """
    L_max = max(item[0].size(1) for item in batch)
    padded_ysr, padded_y = [], []
    for ysr, y in batch:
        pad = L_max - ysr.size(1)
        padded_ysr.append(F.pad(ysr, (0, pad)))
        padded_y.append(F.pad(y,    (0, pad)))
    return torch.stack(padded_ysr, dim=0), torch.stack(padded_y, dim=0)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on", "GPU" if device.type=='cuda' else "CPU")
    SAMPLE_RATE = 16000

    # --- 1) Build the denoising dataset + loaders ----------------------
    full_ds = MultiNoisyFileDataset(
        clean_dir='./irasai/train',
        noisy_dir='./irasai/train/noisy',
        factor=8,
        sample_rate=SAMPLE_RATE,
        target_len=32000,
        debug_crop=False   # 4-second chunks
    )
    ds_len = len(full_ds)
    train_len = int(0.8 * ds_len)
    val_len   = ds_len - train_len
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False)

    # --- 2) Build the SR dataset + loaders -----------------------------
    clean_sr = load_waveforms_from_dir('./irasai/train', SAMPLE_RATE)
    sr_full_ds = SuperResolutionDataset(
        clean_sr,
        sample_rate=SAMPLE_RATE,
        target_len=64000  # same 4-sec window
    )
    sr_train_ds, sr_val_ds = torch.utils.data.random_split(
        sr_full_ds,
        [int(0.8 * len(sr_full_ds)), len(sr_full_ds) - int(0.8 * len(sr_full_ds))]
    )
    sr_train_loader = DataLoader(
        sr_train_ds, batch_size=8, shuffle=True, collate_fn=pad_collate_sr
    )
    sr_val_loader = DataLoader(
        sr_val_ds, batch_size=8, shuffle=False, collate_fn=pad_collate_sr
    )

    # --- 3) Hyperparams & criteria -------------------------------------
    num_epochs = 20
    denoise_criterion = nn.SmoothL1Loss()
    sr_criterion      = nn.L1Loss()

    # --- 4) Ablation loop ------------------------------------------------
    variants = [
        {'use_preconv': True,  'norm': 'layernorm', 'activation': 'silu'},
        {'use_preconv': False, 'norm': 'layernorm', 'activation': 'silu'},
        # {'use_preconv': True,  'norm': 'batchnorm', 'activation': 'relu'},
        {'use_preconv': False, 'norm': 'batchnorm', 'activation': 'relu'},
    ]
    results = {}

    for v in variants:
        prefix   = f"preconv_{v['use_preconv']}_norm_{v['norm']}_act_{v['activation']}"
        ckpt_dir = "checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)

        # 1) Instantiate fresh model / optimizer / scheduler
        model     = VariantATENNuate(**v).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.02)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # 2) Resume from latest checkpoint, if any
        pattern    = os.path.join(ckpt_dir, f"{prefix}_e*.pth")
        candidates = sorted(glob.glob(pattern), key=os.path.getmtime)
        if candidates:
            latest = candidates[-1]
            print(f"🔄 Resuming {prefix} from {latest}")
            ckpt = torch.load(latest, map_location=device)
            model.load_state_dict(      ckpt['model_state_dict'])
            optimizer.load_state_dict(  ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(  ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
        else:
            print(f"Starting {prefix} from scratch")
            start_epoch = 1

        # 3) Training loop
        for epoch in range(start_epoch, num_epochs + 1):
            start = time.time()
            loss = train_epoch(model, train_loader, optimizer,
                   denoise_criterion, device)
                   
            scheduler.step()
            elapsed = time.time() - start

            print(f"[{prefix}] Epoch {epoch:3d} — loss={loss:.4f} elapsed — {elapsed:.1f}s")


            # 4) Periodic checkpoint
            if epoch % 2 == 0 or epoch == num_epochs:
                ckpt_path = os.path.join(ckpt_dir, f"{prefix}_e{epoch}.pth")
                torch.save({
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch':                epoch,
                    'variant':              v,
                }, ckpt_path)
                # print(f"Saved checkpoint {ckpt_path}")

    # 5) Once training is done for this variant, run evaluation & profiling
    den_metrics = eval_epoch(model, val_loader, device)
    prof        = profile_model(model, input_shape=(1,1,SAMPLE_RATE))

    # store into results
    results[prefix] = {
        'denoise': den_metrics,
        'profile': prof,
    }
    print(f"[{prefix}] Denoise metrics:", den_metrics)
    print(f"[{prefix}] Profile:", prof)
    print(f"[{prefix}] Denoise metrics:", den_metrics)
    prof = profile_model(model, input_shape=(1,1,SAMPLE_RATE))
    print(f"[{prefix}] Profile:", prof)

    # --- 5) Final report -----------------------------------------------
    pprint.pprint(results)
