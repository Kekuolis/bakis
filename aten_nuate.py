# aten_nuate.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# (Copy over c2d_zoh_complex, SSMLayerZOH, SSMLayerFFTComplex, PreConv,
#  Resample, CausalPreConv, EncoderBlock, DecoderBlock, and ATENNuate here.)


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

class EncoderBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 resample: int,
                 use_preconv: bool = True,
                 dt: float = 1.0,
                 state_dim: int = 256):
        super().__init__()
        self.use_preconv = use_preconv and in_ch > 1
        if self.use_preconv:
            self.preconv = CausalPreConv(in_ch, kernel_size=3)

        # your SSM layer (with correct dt)
        self.ssm      = SSMLayerFFTComplex(
            in_ch, out_ch,
            state_dim=state_dim,
            dt=dt
        )
        # --- add these two back in! ---
        self.norm     = nn.LayerNorm(out_ch)
        self.resample = Resample(out_ch, out_ch,
                                 factor=resample,
                                 mode='down')
        # activation
        self.act      = nn.SiLU()

    def forward(self, x):
        if self.use_preconv:
            x = self.preconv(x)
        x = self.ssm(x)

        if isinstance(self.norm, nn.BatchNorm1d):
            x = self.norm(x)
            x = self.act(x)
        else:
            x = x.permute(0,2,1)
            x = self.norm(x)
            x = self.act(x)
            x = x.permute(0,2,1)

        x = self.resample(x)
        return x
        
class DecoderBlock(nn.Module):
    def __init__(self,
                    in_ch: int,
                    out_ch: int,
                    resample: int,
                    use_preconv: bool = False,
                    dt: float = 1.0,
                    state_dim: int = 256):
        super().__init__()
        self.ssm = SSMLayerZOH(in_ch, out_ch,
                                state_dim=state_dim,
                                dt=dt)
        # **pass mode='up' here**
        self.resample = Resample(out_ch, out_ch, factor=resample, mode='up')
        self.norm     = nn.LayerNorm(out_ch)
        self.act      = nn.SiLU()

    def forward(self, x):
        # x: (B, C, T)
        x = self.ssm(x)

        # apply norm/activation (special-casing BatchNorm1d vs LayerNorm)
        if isinstance(self.norm, nn.BatchNorm1d):
            x = self.norm(x)
            x = self.act(x)
        else:
            x = x.permute(0,2,1)
            x = self.norm(x)
            x = self.act(x)
            x = x.permute(0,2,1)

        # now upsample back
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
    def __init__(self, in_ch, out_ch, factor, mode='down'):
        super().__init__()
        self.factor = factor
        self.mode   = mode
        self.proj   = nn.Conv1d(in_ch * factor, out_ch, kernel_size=1)

    def forward(self, x):
        B, C, T = x.shape
        if self.mode == 'down':
            pad = (-T) % self.factor
            if pad:
                x = F.pad(x, (0, pad))
                T += pad
            x = x.view(B, C, T // self.factor, self.factor)
            x = x.permute(0, 1, 3, 2).reshape(B, C * self.factor, T // self.factor)
            return self.proj(x)
        else:  # up-sample
            x = x.unsqueeze(-1).expand(-1, -1, -1, self.factor)
            x = x.reshape(B, C * self.factor, T)
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
        self.state_dim = 256
        # Encoder specs: (in→out, resample_factor, use_preconv)
        self.enc_specs = [
            (1,   16, 4, True),
            (16,  32, 4, True),
            (32,  64, 2, True),
            (64,  96, 2, True),
            (96, 128, 2, True),
            (128,256, 2, True),
        ]
        self.dec_specs = [
            (256,128,2),
            (128, 96,2),
            (96, 64, 2),
            (64, 32, 2),
            (32, 16, 4),
            (16,  1, 4),
        ]

        # build layers
        self.encoders = nn.ModuleList([
            EncoderBlock(in_ch=ic,
                         out_ch=oc,
                         resample=r,
                         use_preconv=use_pre,
                         dt=self.dt,
                         state_dim=self.state_dim)
            for ic, oc, r, use_pre in self.enc_specs
        ])
        self.neck = nn.Sequential(
            EncoderBlock(in_ch=256,
                         out_ch=256,
                         resample=1,
                         use_preconv=False,
                         dt=self.dt,
                         state_dim=self.state_dim),
            EncoderBlock(in_ch=256,
                         out_ch=256,
                         resample=1,
                         use_preconv=False,
                         dt=self.dt,
                         state_dim=self.state_dim),
        )
        self.decoders = nn.ModuleList([
            DecoderBlock(in_ch=ic,
                         out_ch=oc,
                         resample=r,
                         dt=self.dt,
                         state_dim=self.state_dim)
            for ic, oc, r in self.dec_specs
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
        T0 = x.size(2)
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
         # finally, if we’re not back to T0, upsample by nearest‐neighbor
        if x.size(2) != T0:
            x = F.interpolate(x, size=T0, mode='nearest')
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
                use_preconv=use_preconv,
                dt=self.dt,
                state_dim=256
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
