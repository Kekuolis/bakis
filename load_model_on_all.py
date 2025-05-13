# Re-run the function setup after code state reset
import os
import torch
import torchaudio
import glob
import re
import gc
#from aten_nuate_lower_memory import VariantATENNuate
from aten_nuate import VariantATENNuate

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

def apply_models_to_directory(input_dir, output_base, ckpt_dir='checkpoints', batch_size=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    variants = [
        {'use_preconv': True,  'norm': 'layernorm', 'activation': 'silu'},
        {'use_preconv': False, 'norm': 'layernorm', 'activation': 'silu'},
        # {'use_preconv': Fal   se, 'norm': 'batchnorm', 'activation': 'relu'},
    ]
    model_sr = 64000
    wav_paths = sorted(glob.glob(os.path.join(input_dir, '*.wav')))
    # before your for-v in variants loop

    wav_paths = sorted(glob.glob(os.path.join(input_dir, '*.wav')))

    for v in variants:
        prefix = f"preconv_{v['use_preconv']}_norm_{v['norm']}_act_{v['activation']}"
        out_dir = os.path.join(output_base, prefix)
        os.makedirs(out_dir, exist_ok=True)

        # 👇 Only keep files not yet processed
        wav_paths_pending = [
            p for p in wav_paths
            if not os.path.exists(os.path.join(out_dir, os.path.basename(p)))
        ]
        print(f"🔍 Processing variant: {prefix} ({len(wav_paths_pending)} files pending)")

        if not wav_paths_pending:
            continue

        model = VariantATENNuate(**v).to(device)
        if not load_latest_checkpoint_if_exists(model, prefix, ckpt_dir, device):
            continue
        model.eval()

        # cache resamplers per orig_sr
        resamplers = {}

        # process only pending files, in batches
        for i in range(0, len(wav_paths_pending), batch_size):
            batch_paths = wav_paths_pending[i : i + batch_size]

            waveforms = []
            lengths = []
            orig_srs = []

            # load & preprocess on CPU→GPU
            for p in batch_paths:
                wav, sr = torchaudio.load(p)               # [C, T]
                # mono
                if wav.size(0) > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                # send to device
                wav = wav.to(device)
                # resample if needed
                if sr != model_sr:
                    if sr not in resamplers:
                        resamplers[sr] = torchaudio.transforms.Resample(
                            orig_freq=sr, new_freq=model_sr
                        ).to(device)
                    wav = resamplers[sr](wav)
                # record
                waveforms.append(wav)
                lengths.append(wav.size(1))
                orig_srs.append(sr)

            # pad to max length in batch
            max_len = max(lengths)
            batch_tensor = torch.zeros(len(waveforms), 1, max_len, device=device)
            for idx, wav in enumerate(waveforms):
                batch_tensor[idx, 0, :lengths[idx]] = wav

            # inference
            with torch.no_grad():
                enhanced = model(batch_tensor)           # [B, 1, L]
            enhanced = enhanced.squeeze(1).cpu()         # [B, L]

            # save each file and cleanup
            for idx, p in enumerate(batch_paths):
                filename = os.path.basename(p)
                out_path = os.path.join(out_dir, filename)
                if os.path.exists(out_path):
                    print(f"⚠️ Skipping existing: {filename}")
                    continue
                # trim to original length
                out_wav = enhanced[idx, :lengths[idx]]
                out_wav = torch.clamp(out_wav, -1.0, 1.0)
                torchaudio.save(out_path, out_wav.unsqueeze(0), model_sr)
                print(f"✔️ Saved: {out_path}")

            # free memory
            del batch_tensor, enhanced, waveforms
            torch.cuda.empty_cache()
            gc.collect()
 
apply_models_to_directory(
    input_dir='./irasai/NOISY',
    output_base='./irasai/test/enhanced_outputs_20_epochs_denoised_64000',
    ckpt_dir='./checkpoints/true_check_points/'
)
