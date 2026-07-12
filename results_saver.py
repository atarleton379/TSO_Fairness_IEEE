import json
import os
from datetime import datetime


def _safe_key(k):
    """Convert a dict key to a JSON-safe string if needed."""
    if isinstance(k, tuple):
        return str(k)
    return k


def _json_safe(value):
    """Convert nested values to JSON-safe Python primitives."""
    if isinstance(value, dict):
        return {_safe_key(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _with_welfare_adjustment(out_dict: dict) -> dict:
    """Attach derived welfare fields for co-optimized reserve/day-ahead outputs."""
    payload = dict(out_dict)
    day_ahead = payload.get("day_ahead_market")
    if isinstance(day_ahead, dict):
        sw = day_ahead.get("social_welfare")
        reserve_cost = day_ahead.get("reserve_procurement_cost")
        if isinstance(sw, (int, float)) and isinstance(reserve_cost, (int, float)):
            payload.setdefault("objective_metadata", {})
            payload["objective_metadata"].update(
                {
                    "objective_orientation": "maximize",
                    "objective_value": sw,
                    "social_welfare_excluding_reserve_cost": sw + reserve_cost,
                    "reserve_cost_sign_in_objective": "subtracted",
                }
            )
    return payload


def save_results(results: dict[str, dict], base_folder: str = "code/results", run_name: str | None = None) -> str:
    """
    Save model output dictionaries to a folder.

    Args:
        results: dict mapping model name (str) to its out_dict (dict)
                 e.g. {"model1": model1.out_dict, "model3": model3.out_dict}
        base_folder: parent folder to create run folder inside
        run_name: if provided, saves to base_folder/run_name (fixed, overwrites each run).
                  if None, creates a timestamped folder (unique per run).

    Returns:
        Path to the folder where results were saved
    """
    folder_name = run_name if run_name else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = os.path.join(base_folder, folder_name)
    os.makedirs(run_folder, exist_ok=True)

    for model_name, out_dict in results.items():
        file_path = os.path.join(run_folder, f"{model_name}.json")
        normalized = _with_welfare_adjustment(_json_safe(out_dict))
        with open(file_path, "w") as f:
            json.dump(normalized, f, indent=4)

    print(f"Results saved to: {run_folder}")
    return run_folder
