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
import speechmetrics
import math
from typing import List, Dict
import time


class SSMLayerFFTComplex(nn.Module):
    def __init__(self, in_ch, out_ch, state_dim=256, dt=1.0):
        """
        in_ch:   number of input channels
        out_ch:  number of output channel
        state_dim: must be even (half complex pairs)
        dt:      time-step (e.g. 1 / sample_rate)
        """
        super().__init__()
        assert state_dim % 2 == 0, "state_dim must be even for conjugate pairs"
        self.in_ch, self.out_ch = in_ch, out_ch
        self.state_dim = state_dim
        self.num_pairs = state_dim // 2
        self.dt = dt

        # continuous-time parameters for each complex pair:
        #   real decay  ← softplus(raw_lambda)
        #   imag freq   = raw_omega
        self.raw_lambda = nn.Parameter(torch.zeros(self.num_pairs))
        self.raw_omega  = nn.Parameter(torch.zeros(self.num_pairs))

        # real-valued B_c and C for each pair (will duplicate to form conjugates)
        self.B_c_pair = nn.Parameter(torch.randn(in_ch, self.num_pairs) *
                                     (1.0 / in_ch**0.5))
        self.C_pair   = nn.Parameter(torch.randn(self.num_pairs, out_ch) *
                                     (1.0 / self.num_pairs**0.5))

    def forward(self, u: torch.Tensor):
        """
        u: (B, in_ch, T) real input
        returns y: (B, out_ch, T) real output
        """
        B, _, T = u.shape
        device = u.device

        # 1) build complex continuous eigenvalues λ = real + i·imag
        real = -F.softplus(self.raw_lambda)
        imag = self.raw_omega
        λ_pair = real + 1j * imag                             # (num_pairs,)
        λ_c = torch.cat([λ_pair, λ_pair.conj()], dim=0).to(device)  # (state_dim,)
         # 2) form full B_c, C by duplicating pairs
        B_c_full = torch.cat([self.B_c_pair, self.B_c_pair], dim=1)  # (in_ch, state_dim)
        C_full   = torch.cat([self.C_pair,   self.C_pair],   dim=0)  # (state_dim, out_ch)

        # 3) discrete ZOH → A_d_diag (complex), B_d (complex)
        A_d_diag, B_d = c2d_zoh_complex(λ_c, B_c_full, self.dt)
        # now that B_d exists (complex), cast C_full to the same dtype
        C_full = C_full.to(device).to(B_d.dtype)

        # h = h_iot.real.permute(2, 0, 1).contiguous()
        # 4) build complex impulse response h_complex over length T via matmul
        #    B_d: (in_ch, state_dim), A_pow: (state_dim, T), C_full: (state_dim, out_ch)
        t = torch.arange(T, device=device).unsqueeze(0)            # (1, T)
        A_pow = A_d_diag.unsqueeze(1) ** t                         # (state_dim, T)
        # expand to multiply B_d * A_pow per time step:
        # M1: (in_ch, state_dim, T)
        M1 = B_d.unsqueeze(-1) * A_pow.unsqueeze(0)
        # now for each in_ch, we have a (T, state_dim) matrix to multiply by C_full
        # reshape M1 to (in_ch, T, state_dim)
        M1_t = M1.permute(0, 2, 1)
        # matmul with C_full: (in_ch, T, out_ch), still complex
        h_iot = torch.matmul(M1_t, C_full)
        # take real part and permute to (out_ch, in_ch, T)
        h = h_iot.real.permute(2, 0, 1).contiguous()

        # 5) FFT-based linear convolution (parallelized)
        # ensure L is a Python int so bit_length() exists
        L_val = int(2 * T - 1)
        # next power-of-two >= L_val
        nfft = 1 << (L_val - 1).bit_length()

        # Option B: use math to next power of two
        # nfft = 2 ** math.ceil(math.log2(L))

        U = torch.fft.rfft(u,   nfft)
        H = torch.fft.rfft(h,   nfft)
        Yf = torch.einsum('bif,oif->bof', U, H)
        y  = torch.fft.irfft(Yf, nfft)[..., :T]
        # 6) at inference (self.training==False), use online complex recurrence
        if not self.training:
            s = torch.zeros(B, self.state_dim, device=device, dtype=torch.cfloat)
            outs = []
            for ti in range(T):
                # cast this slice to complex so matmul matches B_d.dtype
                u_t = u[..., ti].to(B_d.dtype)           # now complex64
                s = A_d_diag.unsqueeze(0) * s  + u_t @ B_d # complex × complex  complex @ complex
                outs.append((s @ C_full).real)          # real part only
            y = torch.stack(outs, dim=-1)


        return y
def c2d_zoh_complex(A_c_diag: torch.Tensor, B_c: torch.Tensor, dt: float):
    """
    Zero-order-hold discretization for complex A_c_diag.
    A_c_diag: (state_dim,) complex-valued
    B_c:       (in_ch, state_dim) real or complex
    dt:        float
    Returns:
      A_d_diag: (state_dim,) complex
      B_d:      (in_ch, state_dim) complex
    """
    A_d_diag = torch.exp(A_c_diag * dt)
    eps = 1e-6
    num = A_d_diag - 1.0
    # safe division: if |λ| small, use dt
    factor = torch.where(
        A_c_diag.abs() > eps,
        num / A_c_diag,
        torch.full_like(num, dt, dtype=num.dtype, device=num.device)
    )
    B_d = B_c * factor.unsqueeze(0)
    return A_d_diag, B_d


class SSMLayerZOH(nn.Module):
    def __init__(self, in_ch, out_ch, state_dim=256, dt=1.0):
        super().__init__()
        self.state_dim = state_dim
        self.dt = dt
        # continuous‐time eigenvalues λᵢ, parameterized to be negative for stability:
        self.raw_lambda = nn.Parameter(torch.zeros(state_dim))
        # continuous‐time B_c, C
        self.B_c = nn.Parameter(torch.randn(in_ch, state_dim) * (1.0 / in_ch**0.5))
        self.C   = nn.Parameter(torch.randn(state_dim, out_ch)  * (1.0 / state_dim**0.5))

    def forward(self, u):
        # u: (B, in_ch, T)
        batch, _, T = u.shape
        device = u.device

        # 1) discretize A_c, B_c → A_d, B_d
        λ_c = -F.softplus(self.raw_lambda)  # continuous λᵢ < 0
        A_d_diag, B_d = c2d_zoh_complex(λ_c, self.B_c, self.dt)  # :contentReference[oaicite:0]{index=0}

        # 2) run recurrence: sₜ = A_d s_{t−1} + B_dᵀ uₜ, yₜ = C sₜ
        s = torch.zeros(batch, self.state_dim, device=device)
        outputs = []
        for t in range(T):
            u_t = u[..., t]                  # (B, in_ch)
            s = s * A_d_diag.unsqueeze(0)   # (B, state_dim)
            s = s + u_t @ B_d               # (B, state_dim)
            y_t = s @ self.C                # (B, out_ch)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=-1)    # (B, out_ch, T)
        return y


# --- Encoder / Decoder Block w/ Causal PreConv ----------------------------
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, resample, use_preconv=True):
        super().__init__()
        self.use_preconv = use_preconv and in_ch > 1
        if self.use_preconv:
            self.preconv = CausalPreConv(in_ch, kernel_size=3)
        self.ssm      = SSMLayerFFTComplex(in_ch, out_ch)  # or whichever SSM you choose
        self.resample = Resample(out_ch, out_ch, resample)
        self.norm     = nn.LayerNorm(out_ch)
        self.act      = nn.SiLU()

    def forward(self, x):
            # x: (B, C, T)
            if self.use_preconv:
                x = self.preconv(x)
            x = self.ssm(x)

            # choose permutation based on norm type
            if isinstance(self.norm, nn.BatchNorm1d):
                # for BN1d we want (B, C, T)
                x = self.norm(x)
                x = self.act(x)
            else:
                # for LayerNorm we want (B, T, C)
                x = x.permute(0, 2, 1)
                x = self.norm(x)
                x = self.act(x)
                x = x.permute(0, 2, 1)

            x = self.resample(x)
            return x



class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, resample, use_preconv=False):
        super().__init__()
        self.ssm      = SSMLayerZOH(in_ch, out_ch)
        self.resample = Resample(out_ch, out_ch, resample)
        # default to LayerNorm; VariantATENNuate may replace this with BatchNorm1d
        self.norm     = nn.LayerNorm(out_ch)
        self.act      = nn.SiLU()

    def forward(self, x):
        # x: (B, C, T)
        x = self.ssm(x)  # → (B, out_ch, T)

        # apply normalization + activation
        if isinstance(self.norm, nn.BatchNorm1d):
            # BatchNorm1d expects (B, C, T)
            x = self.norm(x)
            x = self.act(x)
        else:
            # LayerNorm expects (B, T, C)
            x = x.permute(0, 2, 1)      # → (B, T, C)
            x = self.norm(x)
            x = self.act(x)
            x = x.permute(0, 2, 1)      # → (B, C, T)

        # then resample
        x = self.resample(x)
        return x



# --- PreConv (depthwise) ---------------------------------------------------
class PreConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3,
                              padding=2, padding_mode='zeros',
                              groups=channels)  # depthwise
    def forward(self, x):
        # causal convolution: trim future samples
        x = self.conv(x)
        return x[:, :, :-2]  # remove the extra padding at the end

# --- Resampling layer (squeeze/expand) --------------------------------------

class Resample(nn.Module):
    def __init__(self, in_ch, out_ch, factor):
        super().__init__()
        self.factor = factor
        self.proj = nn.Conv1d(in_ch * factor, out_ch, kernel_size=1)

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        if self.factor > 1:
            # pad time to a multiple of factor
            pad = (-T) % self.factor
            if pad > 0:
                # pad on the right
                x = F.pad(x, (0, pad))
                T = T + pad

            # now safe to reshape
            x = x.view(B, C, T // self.factor, self.factor)
            x = x.permute(0, 1, 3, 2).contiguous()  # (B, C, factor, T')
            x = x.view(B, C * self.factor, T // self.factor)
            return self.proj(x)

        else:
            # upsample: expand time
            x = x.unsqueeze(-1).expand(-1, -1, -1, self.factor)
            x = x.contiguous().view(B, C * self.factor, T)
            return self.proj(x)


# --- Causal PreConv --------------------------------------------------------
class CausalPreConv(nn.Module):
    """
    Depthwise 1D conv that only looks at past samples (causal).
    Kernel size k, so we pad (k-1) zeros on the left.
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=0,        # we pad manually
            groups=channels
        )
    def forward(self, x):
        # x: (B, C, T)
        x = F.pad(x, (self.pad, 0))  # left-pad only
        return self.conv(x)




# --- Full Model w/ Latency Accounting --------------------------------------
class ATENNuate(nn.Module):
    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        # sample period (seconds per sample)
        self.dt = 1.0 / sample_rate

        # Encoder specs: (in→out, resample_factor, use_preconv)
        self.enc_specs = [
            (1,   16, 4, True),
            (16,  32, 4, True),
            (32,  64, 2, True),
            (64,  96, 2, True),
            (96, 128, 2, True),
            (128,256, 2, True),
        ]
        dec_specs = [
            (256,128,2),
            (128, 96,2),
            (96, 64, 2),
            (64, 32, 2),
            (32, 16, 4),
            (16,  1, 4),
        ]

        # build layers
        self.encoders = nn.ModuleList([
            EncoderBlock(ic, oc, r, use_pre)
            for ic, oc, r, use_pre in self.enc_specs
        ])
        self.neck = nn.Sequential(
            EncoderBlock(256, 256, 1, use_preconv=False),
            EncoderBlock(256, 256, 1, use_preconv=False),
        )
        self.decoders = nn.ModuleList([
            DecoderBlock(ic, oc, r)
            for ic, oc, r in dec_specs
        ])
        # for each decoder, a 1×1 conv to align encoder→decoder channels
        self.skip_projs = nn.ModuleList([
            nn.Conv1d(256, 128, 1),  # enc5→dec0
            nn.Conv1d(128,  96, 1),  # enc4→dec1
            nn.Conv1d( 96,  64, 1),  # enc3→dec2
            nn.Conv1d( 64,  32, 1),  # enc2→dec3
            nn.Conv1d( 32,  16, 1),  # enc1→dec4
            nn.Conv1d( 16,   1, 1),  # enc0→dec5 (optional—often you skip this last residual)
        ])

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
        x = self.neck(x)
        for idx, dec in enumerate(self.decoders):
            x = dec(x)
            skip = skips.pop()
            # align channels
            skip = self.skip_projs[idx](skip)
            if skip.size(2) != x.size(2):
                x = F.pad(x, (0, skip.size(2) - x.size(2)))
            x = x + skip
        return x

    def compute_latency(self):
        """
        Compute total lookahead (in seconds) introduced by all causal convs.
        For each encoder PreConv with kernel k and cumulative downsampling f:
          latency += (k-1) * f * dt
        """
        total_samples = 0
        cum_factor = 1
        for in_ch, _, r, use_pre in self.enc_specs:
            if use_pre and in_ch > 1:
                k = 3
                # (k-1) past samples at this layer, each corresponds to cum_factor input samples
                total_samples += (k - 1) * cum_factor
            cum_factor *= r

        return total_samples * self.dt


class VariantATENNuate(ATENNuate):
    def __init__(self,
                 sample_rate: int = 16000,
                 use_preconv: bool = True,
                 norm: str = 'layernorm',    # 'layernorm' or 'batchnorm'
                 activation: str = 'silu'     # 'silu' or 'relu'
                 ):
        super().__init__(sample_rate)
        # override encoder blocks
        for idx, (ic, oc, r, _) in enumerate(self.enc_specs):
            self.encoders[idx] = EncoderBlock(
                in_ch=ic,
                out_ch=oc,
                resample=r,
                use_preconv=use_preconv
            )
            # adjust norm and activation
            block = self.encoders[idx]
            if norm == 'batchnorm':
                block.norm = nn.BatchNorm1d(oc)
            if activation == 'relu':
                block.act = nn.ReLU()
        # override decoder if needed
        for idx, (ic, oc, r) in enumerate([(256,128,2),(128,96,2),(96,64,2),(64,32,2),(32,16,4),(16,1,4)]):
            dec = self.decoders[idx]
            if norm == 'batchnorm':
                dec.norm = nn.BatchNorm1d(oc)
            if activation == 'relu':
                dec.act = nn.ReLU()


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
        target_len=16000   # 4-second chunks
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
    num_epochs = 3
    denoise_criterion = nn.SmoothL1Loss()
    sr_criterion      = nn.L1Loss()

    # --- 4) Ablation loop ------------------------------------------------
    variants = [
        {'use_preconv': True,  'norm': 'layernorm', 'activation': 'silu'},
        {'use_preconv': False, 'norm': 'layernorm', 'activation': 'silu'},
        {'use_preconv': True,  'norm': 'batchnorm', 'activation': 'relu'},
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
            # print(f"Starting {prefix} from scratch")
            start_epoch = 1

        # 3) Training loop
        for epoch in range(start_epoch, num_epochs + 1):
            start = time.time()
            loss, train_si_sdr = train_epoch(model, train_loader, optimizer, denoise_criterion, device)
            scheduler.step()
            elapsed = time.time() - start

            print(f"[{prefix}] Epoch {epoch:3d} — loss={loss:.4f} — SI-SDR={train_si_sdr:.2f} dB — {elapsed:.1f}s")


            # 4) Periodic checkpoint
            if epoch % 5 == 0 or epoch ==    num_epochs:
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
