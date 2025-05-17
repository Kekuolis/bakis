import json
import numpy as np
from typing import Union, Dict, Any

def summarize_pesq_json(
    data_or_path: Union[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Summarize PESQ scores from a JSON file or dict.

    Parameters
    ----------
    data_or_path : str or dict
        - If str: path to a JSON file containing either:
            { "<model>": { "<file>": { "clean_vs_noisy": float, "clean_vs_denoised": float, ... }, ... }, ... }
          or
            { "results": { <as above> }, ... }
        - If dict: the same structure already loaded.

    Returns
    -------
    summary : dict
        {
          "<model>": {
            "clean_vs_noisy": {
              "count": int,
              "mean": float,
              "median": float,
              "std": float,
              "min": float,
              "max": float
            },
            "clean_vs_denoised": { … },
            …
          },
          …
        }
    """
    # Load from file if necessary
    if isinstance(data_or_path, str):
        with open(data_or_path, 'r') as f:
            data = json.load(f)
    else:
        data = data_or_path

    # Drill down if wrapped in a "results" key
    results = data.get("results", data)

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_name, file_dict in results.items():
        # collect lists of scores per metric
        metrics_data: Dict[str, list] = {}
        for fname, scores in file_dict.items():
            if not isinstance(scores, dict):
                continue
            for metric, value in scores.items():
                try:
                    val = float(value)
                except (TypeError, ValueError):
                    continue
                metrics_data.setdefault(metric, []).append(val)

        # compute summary stats
        summary[model_name] = {}
        for metric, values in metrics_data.items():
            arr = np.array(values, dtype=float)
            if arr.size == 0:
                continue
            summary[model_name][metric] = {
                "count": int(arr.size),
                "mean":   float(arr.mean()),
                "median": float(np.median(arr)),
                "std":    float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "min":    float(arr.min()),
                "max":    float(arr.max()),
            }

    return summary

# Example usage:
if __name__ == "__main__":
    # as a file
    stats = summarize_pesq_json("./results/pesq_results_10_epoch_32000pesq_results_10_epoch_64000_gain_eq.json")
    print(json.dumps(stats, indent=2))

    # or on a loaded dict
    # with open("pesq_results_raw.json") as f:
    #     data = json.load(f)
    # stats = summarize_pesq_json(data)
