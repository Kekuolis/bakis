import os
import torch
import torchaudio
import glob
from scipy.signal import wiener


def apply_wiener_filter_to_directory(input_dir, output_dir, model_sr=16000):
    os.makedirs(output_dir, exist_ok=True)
    wav_paths = sorted(glob.glob(os.path.join(input_dir, '*.wav')))

    for wav_path in wav_paths:
        filename = os.path.basename(wav_path)
        out_path = os.path.join(output_dir, filename)

        if os.path.exists(out_path):
            print(f"⚠ Skipping existing: {filename}")
            continue

        wav, sr = torchaudio.load(wav_path)

        # Convert to mono if necessary
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample if necessary
        if sr != model_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=model_sr)
            wav = resampler(wav)

        wav_np = wav.squeeze().numpy()

        # Apply Wiener filter
        filtered_wav = wiener(wav_np)

        # Clamp the audio
        filtered_wav_tensor = torch.clamp(torch.tensor(filtered_wav).unsqueeze(0), -1.0, 1.0)

        # Save output
        torchaudio.save(out_path, filtered_wav_tensor, model_sr)
        print(f"✔ Saved: {out_path}")


apply_wiener_filter_to_directory(
    input_dir='./irasai/test/NOISY/',
    output_dir='/home/kek/Documents/bakis/deep_state/aTENNuate/denoised_samples/preconv_False_norm_layernorm_act_silu'
)
