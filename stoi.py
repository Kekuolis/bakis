import os
import json
from scipy.io import wavfile
from scipy.signal import resample_poly
import numpy as np
from math import gcd
from glob import glob
from pystoi import stoi


def resample_if_needed(signal: np.ndarray, fs_orig: int, fs_target: int) -> np.ndarray:
    """
    Resample the audio signal from fs_orig to fs_target using polyphase filtering.
    """
    if fs_orig == fs_target:
        return signal
    factor = gcd(fs_orig, fs_target)
    up = fs_target // factor
    down = fs_orig // factor
    return resample_poly(signal, up, down)


def match_level(ref: np.ndarray, deg: np.ndarray) -> np.ndarray:
    """
    Match the RMS level of deg to that of ref to avoid level mismatch.
    """
    rms_ref = np.sqrt(np.mean(ref.astype(np.float64)**2))
    rms_deg = np.sqrt(np.mean(deg.astype(np.float64)**2))
    return deg * (rms_ref / (rms_deg + 1e-8))


def evaluate_stoi(ref_path: str, deg_path: str, mode: str = 'normal', fs_target: int = None) -> float:
    """
    Compute the STOI score between a reference and degraded signal.
    Applies resampling, level matching, and returns intelligibility score.
    """
    # Load signals
    fs_ref, ref = wavfile.read(ref_path)
    fs_deg, deg = wavfile.read(deg_path)

    # Ensure mono
    if ref.ndim > 1:
        ref = ref[:, 0]
    if deg.ndim > 1:
        deg = deg[:, 0]

    # Decide target sampling rate
    target_sr = fs_target or fs_ref

    # Resample signals if needed
    ref = resample_if_needed(ref, fs_ref, target_sr)
    deg = resample_if_needed(deg, fs_deg, target_sr)

    # Align lengths
    min_len = min(len(ref), len(deg))
    ref = ref[:min_len]
    deg = deg[:min_len]

    # Normalize to float in [-1, 1]
    def to_float_norm(x: np.ndarray) -> np.ndarray:
        if x.dtype.kind == 'i':
            x = x.astype(np.float64) / np.iinfo(x.dtype).max
        else:
            x = x.astype(np.float64)
            x = np.clip(x, -1.0, 1.0)
        return x

    ref_f = to_float_norm(ref)
    deg_f = to_float_norm(deg)

    # Level match to remove loudness differences
    deg_f = match_level(ref_f, deg_f)

    # Compute STOI score
    extended = True if mode == 'extended' else False
    score = stoi(ref_f, deg_f, target_sr, extended=extended)
    return score


def evaluate_all_pairs(clean_dir: str, noisy_dir: str, denoised_base_dir: str,
                       model_dirs: list, output_json: str, metric: str = 'stoi') -> None:
    """
    Evaluate specified speech quality metric (STOI) on all model outputs.
    """
    results = {}

    for model_name in model_dirs:
        print(f"🔍 Evaluating model: {model_name}")
        model_dir = os.path.join(denoised_base_dir, model_name)
        results[model_name] = {}

        denoised_files = sorted(glob(os.path.join(model_dir, '*.wav')))
        for den_file in denoised_files:
            fname = os.path.basename(den_file)
            utt_id = fname.rsplit('_', 1)[0]
            clean_path = os.path.join(clean_dir, utt_id + '.wav')
            noisy_path = os.path.join(noisy_dir, fname)

            if not os.path.exists(clean_path) or not os.path.exists(noisy_path):
                print(f"⚠️ Skipping {fname}: clean or noisy version not found.")
                continue

            try:
                if metric == 'stoi':
                    score_noisy = evaluate_stoi(clean_path, noisy_path)
                    score_denoised = evaluate_stoi(clean_path, den_file)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")

                results[model_name][fname] = {
                    "clean_vs_noisy": round(score_noisy, 4),
                    "clean_vs_denoised": round(score_denoised, 4)
                }
            except Exception as e:
                print(f"❌ Error evaluating {fname} in {model_name}: {e}")

    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ {metric.upper()} evaluation saved to {output_json}")


if __name__ == "__main__":
    model_dirs = [
        "preconv_True_norm_layernorm_act_silu",
        "preconv_False_norm_layernorm_act_silu"
    ]

    evaluate_all_pairs(
        clean_dir="./irasai/test",
        noisy_dir="./irasai/test/NOISY",
        denoised_base_dir="./irasai/test/enhanced_outputs_20_epoch_denoised_64000",
        model_dirs=model_dirs,
        output_json="stoi_results_levelmatched.json",
        metric='stoi'
    )   