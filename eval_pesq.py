# python3 ./eval_pesq.py /home/kek/Documents/bakis/deep_state/irasai/test/NOISY/R_RD_F3_BG040_a0148_5db.wav ./output_enhanced.wav
from scipy.io import wavfile
from pesq import pesq
import numpy as np
import sys
import os

def evaluate_pesq(ref_path: str, deg_path: str, mode: str = 'wb') -> float:
    """
    Evaluate PESQ between two WAV files.
    
    Args:
        ref_path: Path to reference .wav file
        deg_path: Path to degraded .wav file
        mode: 'wb' for wideband (16kHz), 'nb' for narrowband (8kHz)

    Returns:
        PESQ score (float)
    """
    # Load both files
    fs_ref, ref = wavfile.read(ref_path)
    fs_deg, deg = wavfile.read(deg_path)

    if fs_ref != fs_deg:
        raise ValueError(f"Sample rates do not match: {fs_ref} vs {fs_deg}")

    if fs_ref not in [8000, 16000]:
        raise ValueError("PESQ only supports 8000 or 16000 Hz")

    # Mono only
    if ref.ndim > 1:
        ref = ref[:, 0]
    if deg.ndim > 1:
        deg = deg[:, 0]

    # Truncate to equal length
    min_len = min(len(ref), len(deg))
    ref = ref[:min_len]
    deg = deg[:min_len]

    # Ensure int16
    if ref.dtype != np.int16:
        ref = np.clip(ref, -1.0, 1.0)
        ref = (ref * 32767).astype(np.int16)
    if deg.dtype != np.int16:
        deg = np.clip(deg, -1.0, 1.0)
        deg = (deg * 32767).astype(np.int16)

    # Compute PESQ
    return pesq(fs_ref, ref, deg, mode)

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pesq_eval.py ref.wav deg.wav")
        sys.exit(1)

    ref_file = sys.argv[1]
    deg_file = sys.argv[2]

    if not os.path.exists(ref_file) or not os.path.exists(deg_file):
        print("Error: One or both files do not exist.")
        sys.exit(1)

    score = evaluate_pesq(ref_file, deg_file)
    print(f"PESQ score: {score:.4f}")
