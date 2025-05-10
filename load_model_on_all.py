# Re-run the function setup after code state reset
import os
import torch
import torchaudio
import glob
import re
from aten_nuate import VariantATENNuate

# --- Helper: Load the latest checkpoint if available ---
def load_latest_checkpoint_if_exists(model, prefix, ckpt_dir='checkpoints', device='cpu'):
    pattern = os.path.join(ckpt_dir, f"{prefix}_e*.pth")
    files = glob.glob(pattern)
    if not files:
        print(f"⚠️ Skipping {prefix}: no checkpoint found.")
        return False
    def epoch_of(path): return int(re.search(r"_e(\d+)\.pth$", path).group(1))
    latest = max(files, key=epoch_of)
    ckpt = torch.load(latest, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"✅ Loaded checkpoint: {latest}")
    return True

# --- Process all WAVs with available models ---
def apply_models_to_directory(input_dir, output_base, ckpt_dir='checkpoints'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    variants = [
        {'use_preconv': True,  'norm': 'layernorm', 'activation': 'silu'},
        {'use_preconv': False, 'norm': 'layernorm', 'activation': 'silu'},
        {'use_preconv': True,  'norm': 'batchnorm', 'activation': 'relu'},
        {'use_preconv': False, 'norm': 'batchnorm', 'activation': 'relu'},
    ]
    model_sr = 16000
    wav_paths = glob.glob(os.path.join(input_dir, '*.wav'))

    for v in variants:
        prefix = f"preconv_{v['use_preconv']}_norm_{v['norm']}_act_{v['activation']}"
        print(f"\n🔍 Processing variant: {prefix}")
        model = VariantATENNuate(**v).to(device)
        if not load_latest_checkpoint_if_exists(model, prefix, ckpt_dir, device):
            continue
        model.eval()
        out_dir = os.path.join(output_base, prefix)
        os.makedirs(out_dir, exist_ok=True)

        for wav_path in wav_paths:
            filename = os.path.basename(wav_path)
            out_path = os.path.join(out_dir, filename)
            waveform, sr = torchaudio.load(wav_path)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != model_sr:
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=model_sr)(waveform)
            waveform = waveform.unsqueeze(0).to(device)
            with torch.no_grad():
                enhanced = model(waveform)
            enhanced = enhanced.squeeze(0).cpu()
            enhanced = torch.clamp(enhanced, -1.0, 1.0)
            torchaudio.save(out_path, enhanced, model_sr)
            print(f"✔️ Saved: {out_path}")
            
apply_models_to_directory(
    input_dir='/home/kek/Documents/bakis/deep_state/irasai/test/NOISY',
    output_base='./irasai/test/enhanced_outputs',
    ckpt_dir='checkpoints'
)