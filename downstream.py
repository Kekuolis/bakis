import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import List, Dict, Tuple
import glob
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from pesq import pesq as pesq_eval   # pip install pesq


# --- Utility: MACs and latency profiler -----------------------------------
def profile_model(model: nn.Module, input_shape: tuple, sample_rate: int = 16000) -> Dict[str, float]:
    """
    Profiles MACs and computes algorithmic latency for a model.
    Returns: {"macs_million": ..., "latency_ms": ...}
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        raise ImportError("Please install fvcore for MACs profiling: pip install fvcore")

    # make a dummy input on the same device as the model
    device = next(model.parameters()).device
    dummy = torch.randn(*input_shape, device=device)

    # Run the flop counter on model + dummy, now both on the same device
    flops = FlopCountAnalysis(model, dummy)
    macs   = flops.total() / 1e6  # in millions

    # compute latency (this part doesn’t involve tensors)
    latency = 1000.0 * getattr(model, 'compute_latency', lambda sr: 0)(sample_rate)

    return {"macs_million": macs, "latency_ms": latency}
# --- Ablation variants setup ----------------------------------------------

# --- Downstream tasks: super-resolution & de-quantization ----------------

def mu_law_quantize(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Mu-law companding and quantization (device-safe)."""
    # integer μ
    mu = 2**bits - 1
    # make μ into a tensor on the same device & dtype as x
    mu_tensor = torch.tensor(mu, dtype=x.dtype, device=x.device)

    # μ-law encode
    #   x_mu = sign(x) * log1p(μ * |x|) / log1p(μ)
    x_mu = torch.sign(x) * torch.log1p(mu_tensor * x.abs()) / torch.log1p(mu_tensor)

    # quantize to [−1,1]
    #   (x_mu+1)/2 ∈ [0,1]; scale by μ, floor, then map back
    x_q = ((x_mu + 1) / 2 * mu_tensor + 0.5).floor() / mu_tensor * 2 - 1

    return x_q


class SuperResolutionDataset(torch.utils.data.Dataset):
    """Dataset that simulates 4 kHz 4-bit μ-law inputs and provides 16 kHz targets,
       cropping/padding to a fixed length if desired."""
    def __init__(self,
                 clean_waveforms: List[torch.Tensor],
                 sample_rate: int = 16000,
                 target_len: int = None):
        self.clean       = clean_waveforms
        self.sample_rate = sample_rate
        self.target_len  = target_len
        # build the resamplers based on the provided sample_rate
        self.down = torchaudio.transforms.Resample(sample_rate, 4000)
        self.up   = torchaudio.transforms.Resample(4000, sample_rate)

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        y = self.clean[idx]                          # (1, L)
        y4 = self.down(y)                            # (1, L4)
        y4q = mu_law_quantize(y4, bits=4)            # (1, L4)
        y_sr = self.up(y4q)                          # (1, L')

        # If target_len was specified, crop or pad to exactly that length:
        if self.target_len is not None:
            L = y_sr.size(1)
            T = self.target_len
            if L > T:
                start = torch.randint(0, L-T+1, (1,)).item()
                y_sr = y_sr[:, start:start+T]
                y    = y   [:, start:start+T]
            elif L < T:
                pad = T - L
                y_sr = F.pad(y_sr, (0, pad))
                y    = F.pad(y,    (0, pad))

        return y_sr, y


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
    return (
        DataLoader(ds, batch_size, shuffle=True,  collate_fn=pad_collate_sr),
        DataLoader(ds, batch_size, shuffle=False, collate_fn=pad_collate_sr),
    )
class MultiNoisyFileDataset(Dataset):
    """
    For each clean WAV in clean_dir, finds all noisy WAVs in noisy_dir
    whose stem is clean_stem + '_' + suffix (e.g. '...a0362_5db.wav').
    Flattens into (noisy_path, clean_path) pairs.
    """
    def __init__(self, clean_dir, noisy_dir,
                    factor: int, sample_rate: int = 16000,
                    target_len: int = None, debug_crop=False):
        """
        clean_dir: directory with clean files like L_RA_F4_DK015_a0362.wav
        noisy_dir: directory with noisy like L_RA_F4_DK015_a0362_5db.wav, etc
        factor:    expected number of noisy variants per clean
        sample_rate: resample all to this rate
        target_len: if not None, will crop/pad each to this many samples
        """
        self.sample_rate = sample_rate
        self.target_len  = target_len
        self.debug_crop  = debug_crop

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
        
        self.target_len = target_len

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
        # print(f"Pairing: {stem_clean}  ←  {stem_noisy}")
        noisy_path, clean_path = self.pairs[idx]
        noisy = self._load_and_resample(noisy_path)
        clean = self._load_and_resample(clean_path)

        # optional: crop/pad to fixed length
        if self.target_len is not None:
            T = self.target_len
            L = noisy.size(1)
            if L > T:
                start = torch.randint(0, L - T + 1, (1,)).item()
                if self.debug_crop:
                    stem = os.path.splitext(os.path.basename(clean_path))[0]
                    print(f"[DEBUG] {stem}: crop start={start}")
                noisy = noisy[:, start:start+T]
                clean = clean[:, start:start+T]
            elif L < T:
                pad = T - L
                noisy = F.pad(noisy, (0, pad))
                clean = F.pad(clean, (0, pad))

        return noisy, clean

# 2) TRAIN / EVAL LOOPS -----------------------------------------------------

def batch_si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute SI-SDR per example in a batch.
    est, ref: (B, T) torch tensors on the same device.
    Returns: (B,) SI-SDR in dB.
    """
    # zero-mean
    est_zm = est - est.mean(dim=1, keepdim=True)
    ref_zm = ref - ref.mean(dim=1, keepdim=True)
    # projection of est onto ref
    # α = (est_zm·ref_zm) / ||ref_zm||^2
    alpha = torch.sum(est_zm * ref_zm, dim=1, keepdim=True) \
            / (torch.sum(ref_zm.pow(2), dim=1, keepdim=True) + eps)
    proj  = alpha * ref_zm
    noise = est_zm - proj
    # ratio of energies
    si_sdr = 10 * torch.log10(
        (torch.sum(proj.pow(2), dim=1) + eps)
        / (torch.sum(noise.pow(2), dim=1) + eps)
    )
    return si_sdr  # shape (B,)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_examples = 0.0, 0

    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()

        out = model(noisy)
        loss = criterion(out, clean)  # e.g. criterion = nn.L1Loss() or nn.SmoothL1Loss()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * noisy.size(0)
        total_examples += noisy.size(0)

    return total_loss / total_examples
def compute_si_sdr(est: np.ndarray, ref: np.ndarray, eps=1e-8) -> float:
    """
    est, ref: 1D numpy arrays of equal length.
    """
    # zero-mean
    est_zm = est - est.mean()
    ref_zm = ref - ref.mean()
    # projection
    ref_energy = np.sum(ref_zm ** 2) + eps
    proj = (np.sum(est_zm * ref_zm) / ref_energy) * ref_zm
    noise = est_zm - proj
    return 10 * np.log10((np.sum(proj ** 2) + eps) / (np.sum(noise ** 2) + eps))
@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    pesq_scores, si_sdr_scores = [], []
    from pesq import pesq as pesq_eval, NoUtterancesError

    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        est = model(noisy)

        refs =  clean.squeeze(1).cpu().numpy()  # (B, T_ref)
        outs =  est .squeeze(1).cpu().numpy()  # (B, T_out)

        for ref_np, out_np in zip(refs, outs):
            # 1) align lengths
            L = min(len(ref_np), len(out_np))
            ref_np = ref_np[:L]
            out_np = out_np[:L]

            # 2) SI-SDR
            si = compute_si_sdr(out_np, ref_np)
            si_sdr_scores.append(si)

            # 3) PESQ with fallback
            try:
                p = pesq_eval(16000, ref_np, out_np, 'wb')
            except NoUtterancesError:
                p = float('nan')
            pesq_scores.append(p)

    return {
        'pesq': float(np.nanmean(pesq_scores)),
        'si_sdr': float(np.mean(si_sdr_scores))
    }

# 3) DATASET: Windowed noisy dataset --------------------------------------
class WindowedNoisyDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir,
                 sample_rate=16000, window_len=16000, hop=None):
        import glob, os
        self.window_len = window_len
        self.hop = hop or window_len  # non-overlapping by default
        self.entries = []
        clean_files = sorted(glob.glob(f"{clean_dir}/*.wav"))
        for cpath in clean_files:
            npath = os.path.join(noisy_dir, os.path.basename(cpath))
            waveform, sr = torchaudio.load(cpath)
            assert sr == sample_rate
            total = waveform.size(1)
            # slide a window through the file
            for start in range(0, total, self.hop):
                self.entries.append((cpath, npath, start))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        cpath, npath, start = self.entries[idx]
        clean, _ = torchaudio.load(cpath, frame_offset=start,
                                   num_frames=self.window_len)
        noisy, _ = torchaudio.load(npath, frame_offset=start,
                                   num_frames=self.window_len)
        # pad if file tail is shorter than window_len
        L = clean.size(1)
        if L < self.window_len:
            pad = self.window_len - L
            import torch.nn.functional as F
            clean = F.pad(clean, (0, pad))
            noisy = F.pad(noisy, (0, pad))
        return noisy, clean(tinklas) 