import json
import pandas as pd

# Load STOI results from file
with open("stoi_results.json", "r") as f:
    results = json.load(f)

# Convert to DataFrame
rows = []
for model, files in results.items():
    for fname, vals in files.items():
        snr_part = fname.rsplit("_", 1)[-1].replace(".wav", "")
        try:
            snr = int(snr_part.replace("db", ""))
        except ValueError:
            continue
        rows.append({
            "Model": model,
            "File": fname,
            "SNR (dB)": snr,
            "STOI (Noisy)": vals["clean_vs_noisy"],
            "STOI (Denoised)": vals["clean_vs_denoised"]
        })

df = pd.DataFrame(rows)
print(df.groupby(["Model", "SNR (dB)"])[["STOI (Noisy)", "STOI (Denoised)"]].mean().round(4))
