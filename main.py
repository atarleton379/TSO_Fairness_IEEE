import os
import re
import sys

import gurobipy as gp

from input import InputHandler
from model_helpers import load_input_file
from Day_ahead_zonal_model import DayAheadModel
from Standard_reserve_model import Standard_Reserve_Model
from PF_Standard_reserve_model import PF_Standard_Reserve_Model
from MMF_Standard_reserve_model import MMF_Standard_Reserve_Model
from results_saver import save_results
from model_helpers import update_atc, ATC_calc, reference_incident_by_zone
from visualizer import visualize_all_reserve_models


def _coerce_atc_dict(raw):
    """Convert scenario ATC mappings into directional tuples keyed by (u, v)."""
    if not raw:
        return {}

    out = {}
    for key, value in raw.items():
        if isinstance(key, tuple):
            u, v = key
        elif isinstance(key, list):
            u, v = key
        elif isinstance(key, str):
            cleaned = key.strip()
            if cleaned.startswith("(") and cleaned.endswith(")"):
                cleaned = cleaned[1:-1]
            parts = re.split(r"[,\-\>]+", cleaned)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
        else:
            continue

        try:
            out[(int(u), int(v))] = float(value)
        except ValueError:
            continue
    return out


def _build_day_ahead_placeholder():
    """Minimal placeholder output for toy runs that skip the day-ahead solve."""
    return {
        "cross_zonal_flows": {},
        "social_welfare": 0.0,
        "solve_time_seconds": 0.0,
        "zone_1_price": 0.0,
    }


class DummyDA:
    def __init__(self):
        self.out_dict = _build_day_ahead_placeholder()


def build_input_handler(scenario_file: str | None):
    """
    Build the input object.

    - No scenario file: use InputHandler defaults.
    - Scenario file: load JSON and override matching/default attributes.
    """
    inp_hndl = InputHandler()

    if not scenario_file:
        return inp_hndl

    scenario = load_input_file(scenario_file)
    for key, value in vars(scenario).items():
        setattr(inp_hndl, key, value)

    if hasattr(inp_hndl, "zones") and isinstance(inp_hndl.zones, dict):
        inp_hndl.zones = {int(k): v for k, v in inp_hndl.zones.items()}

    return inp_hndl


def main():
    gp.setParam("OutputFlag", 0)
    gp.setParam("LogToConsole", 0)
    gp.setParam("LicenseID", 2837159)

    scenario_file = "toy_scenarios/toy_3zone.json"
    hour = 12
    sensitivity = 1
    results_dir = "code/results"

    inp_hndl = build_input_handler(scenario_file)

    if getattr(inp_hndl, "skip_day_ahead", False):
        day_ahead_zonal = DummyDA()
        atc = _coerce_atc_dict(getattr(inp_hndl, "atc", {}))
        if not atc:
            atc = ATC_calc(inp_hndl.zones, inp_hndl.lines, symmetric=False)
        congested_atc = atc
    else:
        day_ahead_zonal = DayAheadModel(hour, inp_hndl, sensitivity)
        atc = ATC_calc(inp_hndl.zones, inp_hndl.lines, symmetric=False)
        congested_atc = update_atc(atc, day_ahead_zonal.out_dict)

    ri_by_zone = reference_incident_by_zone(inp_hndl)

    std_reserve = Standard_Reserve_Model(hour, inp_hndl, congested_atc, ri_by_zone)
    std_reserve_out = std_reserve.solve()

    pf_reserve = PF_Standard_Reserve_Model(hour, inp_hndl, congested_atc, ri_by_zone)
    pf_reserve_out = pf_reserve.solve()

    mmf_reserve = MMF_Standard_Reserve_Model(hour, inp_hndl, congested_atc, ri_by_zone)
    mmf_reserve_out = mmf_reserve.solve()

    run_folder = save_results(
        {
            "day_ahead_zonal": day_ahead_zonal.out_dict,
            "atc": atc,
            "congested_atc": congested_atc,
            "standard_reserve": std_reserve_out,
            "pf_standard_reserve": pf_reserve_out,
            "mmf_standard_reserve": mmf_reserve_out,
        },
        base_folder=results_dir,
    )

    visualize_all_reserve_models(
        inp_hndl,
        {
            "day_ahead_zonal": day_ahead_zonal.out_dict,
            "standard_reserve": std_reserve_out,
            "pf_standard_reserve": pf_reserve_out,
            "mmf_standard_reserve": mmf_reserve_out,
            "atc": atc,
            "congested_atc": congested_atc,
        },
        output_dir=run_folder,
    )


if __name__ == "__main__":
    main()