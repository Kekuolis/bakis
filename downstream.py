import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import List, Dict, Tuple
import glob
import os
from main import ATENNuate
from torch.utils.data import Dataset, DataLoader

# --- Utility: MACs and latency profiler -----------------------------------
def profile_model(model: nn.Module, input_shape: tuple, sample_rate: int = 16000) -> Dict[str, float]:
    """
    Profiles MACs and computes algorithmic latency for a model.
    Returns: {"macs": ..., "latency_ms": ...}
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        raise ImportError("Please install fvcore for MACs profiling: pip install fvcore")
    dummy = torch.randn(*input_shape)
    flops = FlopCountAnalysis(model, dummy)
    macs = flops.total() / 1e6  # in millions
    latency = 1000.0 * getattr(model, 'compute_latency', lambda: 0)(sample_rate=sample_rate)
    return {"macs_million": macs, "latency_ms": latency}

# --- Ablation variants setup ----------------------------------------------

# --- Downstream tasks: super-resolution & de-quantization ----------------

def mu_law_quantize(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Mu-law companding and quantization."""
    mu = 2**bits - 1
    # mu-law encode
    x_mu = torch.sign(x) * torch.log1p(mu * x.abs()) / torch.log1p(torch.tensor(mu, dtype=x.dtype))
    # quantize
    x_q = ((x_mu + 1) / 2 * mu + 0.5).floor() / mu * 2 - 1
    return x_q

class SuperResolutionDataset(torch.utils.data.Dataset):
    """Dataset that simulates 4kHz 4-bit mu-law inputs and provides target at 16kHz."""
    def __init__(self, clean_waveforms: List[torch.Tensor]):
        self.clean = clean_waveforms
        self.down = torchaudio.transforms.Resample(16000, 4000)
        self.up = torchaudio.transforms.Resample(4000, 16000)

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        y = self.clean[idx]
        y4 = self.down(y)
        y4q = mu_law_quantize(y4, bits=4)
        y_sr = self.up(y4q)
        return y_sr.unsqueeze(0), y.unsqueeze(0)

# --- Example: instantiate and profile variants ----------------------------
if __name__ == '__main__':
    variants = [
        {'use_preconv': True,  'norm': 'layernorm', 'activation': 'silu'},
        {'use_preconv': False, 'norm': 'layernorm', 'activation': 'silu'},
        {'use_preconv': True,  'norm': 'batchnorm', 'activation': 'relu'},
        {'use_preconv': False, 'norm': 'batchnorm', 'activation': 'relu'},
    ]
    results = {}
    for v in variants:
        key = f"preconv={v['use_preconv']}, {v['norm']}, {v['activation']}"
        model = VariantATENNuate(**v)
        prof = profile_model(model, input_shape=(1,1,16000), sample_rate=16000)
        results[key] = prof
    print(results)



# 1) DATALOADERS ------

def make_denoise_loaders(clean_set: List[torch.Tensor],
                         noisy_set: List[torch.Tensor],
                         batch_size: int = 16):
    """Assumes clean_set[i] and noisy_set[i] are paired waveforms."""
    class DenoiseDataset(torch.utils.data.Dataset):
        def __init__(self, clean, noisy):
            self.clean, self.noisy = clean, noisy
        def __len__(self): return len(self.clean)
        def __getitem__(self, i):
            return self.noisy[i].unsqueeze(0), self.clean[i].unsqueeze(0)

    ds = DenoiseDataset(clean_set, noisy_set)
    return DataLoader(ds, batch_size, shuffle=True), DataLoader(ds, batch_size, shuffle=False)

def make_sr_loader(clean_set: List[torch.Tensor], batch_size: int = 16):
    ds = SuperResolutionDataset(clean_set)
    return DataLoader(ds, batch_size, shuffle=True), DataLoader(ds, batch_size, shuffle=False)

class MultiNoisyFileDataset(Dataset):
    """
    For each clean WAV in clean_dir, finds all noisy WAVs in noisy_dir
    whose stem is clean_stem + '_' + suffix (e.g. '...a0362_5db.wav').
    Flattens into (noisy_path, clean_path) pairs.
    """
    def __init__(self,
                 clean_dir: str,
                 noisy_dir: str,
                 factor: int = 8,
                 sample_rate: int = 16000,
                 target_len: int = None):
        """
        clean_dir: directory with clean files like L_RA_F4_DK015_a0362.wav
        noisy_dir: directory with noisy like L_RA_F4_DK015_a0362_5db.wav, etc
        factor:    expected number of noisy variants per clean
        sample_rate: resample all to this rate
        target_len: if not None, will crop/pad each to this many samples
        """
        self.sample_rate = sample_rate
        self.target_len  = target_len

        # 1) map clean stems → full path
        clean_paths = sorted(glob.glob(os.path.join(clean_dir, '*.wav')))
        self.clean_map = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in clean_paths
        }

        # 2) group noisy paths by matching stem
        self.pairs = []
        noisy_paths = sorted(glob.glob(os.path.join(noisy_dir, '*.wav')))
        # build a map clean_stem → list of noisy paths
        grouping = {}
        for p in noisy_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            # strip the last “_5db” (or whatever suffix) to get the clean key
            clean_key = "_".join(stem.split('_')[:-1])
            grouping.setdefault(clean_key, []).append(p)

        # 3) for each clean, ensure you have exactly `factor` noisy variants
        for clean_key, clean_path in self.clean_map.items():
            noisy_list = grouping.get(clean_key, [])
            if len(noisy_list) != factor:
                raise ValueError(f"{clean_key!r}: expected {factor} noisy files, found {len(noisy_list)}")
            for noisy_path in noisy_list:
                self.pairs.append((noisy_path, clean_path))

    def __len__(self):
        return len(self.pairs)

    def _load_and_resample(self, path):
        wav, sr = torchaudio.load(path)    # (C, T)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.transforms.Resample(sr, self.sample_rate)(wav)
        return wav  # (1, T)

    def __getitem__(self, idx):

        noisy_path, clean_path = self.pairs[idx]
        stem_noisy = os.path.basename(noisy_path).split('.')[0]
        stem_clean = os.path.basename(clean_path).split('.')[0]
        print(f"Pairing: {stem_clean}  ←  {stem_noisy}")
        noisy = self._load_and_resample(noisy_path)
        clean = self._load_and_resample(clean_path)
        noisy_path, clean_path = self.pairs[idx]
        noisy = self._load_and_resample(noisy_path)
        clean = self._load_and_resample(clean_path)

        # optional: crop/pad to fixed length
        if self.target_len is not None:
            T = self.target_len
            L = noisy.size(1)
            if L > T:
                start = torch.randint(0, L - T + 1, (1,)).item()
                noisy = noisy[:, start:start+T]
                clean = clean[:, start:start+T]
            elif L < T:
                pad = T - L
                noisy = F.pad(noisy, (0, pad))
                clean = F.pad(clean, (0, pad))

        return noisy, clean

def make_multi_noisy_loaders(clean_list: List[torch.Tensor],
                             noisy_list: List[torch.Tensor],
                             factor: int = 6,
                             batch_size: int = 16,
                             shuffle_train: bool = True):
    """
    Returns train and val loaders for multi-noisy pairing.
    Splits 80/20 by default.
    """
    ds = MultiNoisyDenoiseDataset(clean_list, noisy_list, factor)
    n = len(ds)
    split = int(0.8 * n)
    train_ds, val_ds = torch.utils.data.random_split(ds, [split, n - split])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
# 2) TRAIN / EVAL LOOPS -----------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        out = model(noisy)
        loss = criterion(out, clean)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * noisy.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    import speechmetrics  # pip install speechmetrics
    meter = speechmetrics.load(['pesq', 'si_sdr'])
    all_preds, all_refs = [], []
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        pred = model(noisy)
        all_preds.append(pred.cpu())
        all_refs.append(clean.cpu())
    preds = torch.cat(all_preds, dim=0)   # (N,1,T)
    refs  = torch.cat(all_refs,  dim=0)
    # reshape to (N, T)
    meter_results = meter(preds.squeeze(1), refs.squeeze(1), rate=16000)
    return meter_resul



