import os
import json
from scipy.io import wavfile
from pesq import pesq
import numpy as np
from glob import glob

def evaluate_pesq(ref_path: str, deg_path: str, mode: str = 'wb') -> float:
    fs_ref, ref = wavfile.read(ref_path)
    fs_deg, deg = wavfile.read(deg_path)

    # Ensure mono
    if ref.ndim > 1:
        ref = ref[:, 0]
    if deg.ndim > 1:
        deg = deg[:, 0]

    min_len = min(len(ref), len(deg))
    ref = ref[:min_len]
    deg = deg[:min_len]

    if ref.dtype != np.int16:
        ref = np.clip(ref, -1.0, 1.0)
        ref = (ref * 32767).astype(np.int16)
    if deg.dtype != np.int16:
        deg = np.clip(deg, -1.0, 1.0)
        deg = (deg * 32767).astype(np.int16)

    # Compute PESQ score
    return pesq(fs_ref, ref, deg, mode)


def evaluate_all_pairs(clean_dir, noisy_dir, denoised_base_dir, model_dirs, output_json):
    results = {}

    for model_name in model_dirs:
        print(f"🔍 Evaluating model: {model_name}")
        model_dir = os.path.join(denoised_base_dir, model_name)
        results[model_name] = {}
        print(f"🔍 Evaluating model: {model_name}")
        print(f"[DEBUG] model_dir exists? {os.path.isdir(model_dir)}")
        denoised_files = sorted(glob(os.path.join(model_dir, '*.wav')))
        print(f"[DEBUG] found {len(denoised_files)} .wav files in {model_dir!r}")

        denoised_files = sorted(glob(os.path.join(model_dir, '*.wav')))
        for den_file in denoised_files:
            fname = os.path.basename(den_file)
            utt_id = fname.rsplit('_', 1)[0]
            clean_path = os.path.join(clean_dir, utt_id + '.wav')
            noisy_path = os.path.join(noisy_dir, fname)
            if not os.path.exists(clean_path):
                print(f"⚠️ Missing clean file for {fname}: expected {clean_path}")
                continue
            if not os.path.exists(noisy_path):
                print(f"⚠️ Missing noisy file for {fname}: expected {noisy_path}")
                continue

            try:
                pesq_clean_noisy = evaluate_pesq(clean_path, noisy_path)
                pesq_clean_denoised = evaluate_pesq(clean_path, den_file)
                results[model_name][fname] = {
                    "clean_vs_noisy": round(pesq_clean_noisy, 4),
                    "clean_vs_denoised": round(pesq_clean_denoised, 4)
                }
            except Exception as e:
                print(f"❌ Error evaluating {fname} in {model_name}: {e}")

    # Save results to JSON
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ PESQ evaluation saved to {output_json}")


if __name__ == "__main__":
    # Define which models to run
    model_dirs = [
        # "preconv_True_norm_layernorm_act_relu",
        # "preconv_False_norm_layernorm_act_relu",
        "preconv_True_norm_layernorm_act_silu",
        "preconv_False_norm_layernorm_act_silu",
    ]

    evaluate_all_pairs(
        clean_dir="./irasai/test",
        noisy_dir="./irasai/test/NOISY",
        denoised_base_dir="/home/kek/Documents/bakis/deep_state/irasai/test/enhanced_outputs_20_epoch_denoised_16000_no_dc",
        model_dirs=model_dirs,
        output_json="16000_pesq_eval_20ep_nrml.json",
    )
