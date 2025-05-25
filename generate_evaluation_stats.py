import json
import numpy as np
from typing import Union, Dict, Any
import re

import statistics
from typing import Union, Dict, Any
import math

def summarize_pesq_json_detailed(
    data_or_path: Union[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Summarize PESQ scores per noise level (5–40 dB), deterministically, with full stats.
    """
    # 1) load JSON
    if isinstance(data_or_path, str):
        with open(data_or_path, 'r') as f:
            data = json.load(f)
    else:
        data = data_or_path
    results = data.get("results", data)

    noise_re = re.compile(r"(\d+)[dD][bB]")

    summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    # 2) sort model names for deterministic iteration
    for model_name in sorted(results.keys()):
        file_dict = results[model_name]
        # collects: metrics_data[noise_db][metric] = list of floats
        metrics_data: Dict[int, Dict[str, list]] = {}

        # 3) sort filenames so the append‐order never changes
        for fname in sorted(file_dict.keys()):
            scores = file_dict[fname]
            if not isinstance(scores, dict):
                continue

            m = noise_re.search(fname)
            if not m:
                continue
            noise_db = int(m.group(1))
            if not (5 <= noise_db <= 40):
                continue

            bucket = metrics_data.setdefault(noise_db, {})
            # 4) sort metric keys too (optional but extra-safe)
            for metric in sorted(scores.keys()):
                try:
                    val = float(scores[metric])
                except (TypeError, ValueError):
                    continue
                bucket.setdefault(metric, []).append(val)

        # 5) compute stats with math.fsum / statistics for order‐invariant sums
        summary[model_name] = {}
        for noise_db in sorted(metrics_data.keys()):
            key = f"{noise_db}dB"
            summary[model_name][key] = {}
            for metric, vals in metrics_data[noise_db].items():
                cnt = len(vals)
                if cnt == 0:
                    continue
                mean_   = statistics.mean(vals)            # uses math.fsum
                median_ = statistics.median(vals)
                std_    = statistics.stdev(vals) if cnt > 1 else 0.0
                min_    = min(vals)
                max_    = max(vals)

                summary[model_name][key][metric] = {
                    "count":  cnt,
                    "mean":   mean_,
                    # "median": median_,
                    # "std":    std_,
                    # "min":    min_,
                    # "max":    max_,
                }

    return summary
# Example usage:
if __name__ == "__main__":
    # as a file
    stats = summarize_pesq_json_detailed("./pesql_eval_100ep_trainable_0.005_0.002w.json") # output > stats_20e_nrml_16000.json
    print(json.dumps(stats, indent=2))

    # or on a loaded dict
    # with open("pesq_results_raw.json") as f:
    #     data = json.load(f)
    # stats = summarize_pesq_json(data)
