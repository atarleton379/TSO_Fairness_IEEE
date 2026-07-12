import json
from pathlib import Path
from types import SimpleNamespace
from input import InputHandler


def load_input_file(input_file: str):
    """Load a JSON input file and return a SimpleNamespace."""
    path = Path(input_file)
    if not path.is_absolute():
        candidates = [
            Path.cwd() / path,
            Path(__file__).resolve().parent / path,
            Path(__file__).resolve().parent / "toy_scenarios" / path,
            Path.cwd() / "toy_scenarios" / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return SimpleNamespace(**data)

def reference_incident_by_zone(In):
    """
    Return a dict mapping each zone to its reference incident, defined as
    the largest P_max among generators and wind farms in that zone.
    """
    zones = In.zones
    generators = In.generators
    wind_farms = In.wind_farms

    # Build node -> zone map and validate unique assignment.
    node_to_zone = {}
    for zone_id, zone_nodes in zones.items():
        for node in zone_nodes:
            if node in node_to_zone:
                raise ValueError(f"Node {node} appears in multiple zones.")
            node_to_zone[node] = zone_id

    ri_by_zone = {zone_id: 0.0 for zone_id in zones}

    def scan_assets(assets, label):
        for asset_id, asset in assets.items():
            if "node" not in asset or "P_max" not in asset:
                raise ValueError(
                    f"{label} {asset_id} must contain 'node' and 'P_max'."
                )

            node = asset["node"]
            p_max = float(asset["P_max"])
            zone_id = node_to_zone.get(node)

            if zone_id is None:
                raise ValueError(
                    f"{label} {asset_id} uses node {node}, which is not in zones."
                )

            if p_max > ri_by_zone[zone_id]:
                ri_by_zone[zone_id] = p_max

    scan_assets(generators, "Generator")
    scan_assets(wind_farms, "Wind farm")

    return ri_by_zone

    
def ATC_calc(zones, lines, symmetric=False):
    """
    Returns a dictionary with total capacity between zones.

    If symmetric=True:
        keys are ordered tuples (min_zone, max_zone), e.g. atc[(1, 3)]
    If symmetric=False:
        keys are directional tuples (z_from, z_to), e.g. atc[(1, 3)] and atc[(3, 1)]
    """
    # Map each node to its zone for O(1) zone lookup per line endpoint.
    node_to_zone = {}
    for z, node_list in zones.items():
        for n in node_list:
            if n in node_to_zone:
                raise ValueError(f"Node {n} appears in multiple zones.")
            node_to_zone[n] = z

    atc = {}

    # Initialize all zone combinations so missing links return 0 instead of KeyError.
    zone_ids = sorted(zones.keys())
    if symmetric:
        for i, z1 in enumerate(zone_ids):
            for z2 in zone_ids[i + 1:]:
                atc[(z1, z2)] = 0.0
    else:
        for z1 in zone_ids:
            for z2 in zone_ids:
                if z1 != z2:
                    atc[(z1, z2)] = 0.0

    # Aggregate line capacities if they connect two different zones.
    for _, d in lines.items():
        u = d["from"]
        v = d["to"]
        cap = float(d["capacity"])

        zu = node_to_zone.get(u)
        zv = node_to_zone.get(v)

        if zu is None or zv is None:
            raise ValueError(f"Line endpoint outside zone map: ({u}, {v})")

        if zu == zv:
            continue

        if symmetric:
            key = (zu, zv) if zu < zv else (zv, zu)
            atc[key] += cap
        else:
            atc[(zu, zv)] += cap
            atc[(zv, zu)] += cap

    return atc

def update_atc(atc, out_dict, down_regulation=False):
    """Return directional ATC values after day-ahead flows are applied."""
    da_flows = out_dict.get("cross_zonal_flows", {})
    congested_atc = {}

    for (u, v), cap in atc.items():
        if (u, v) in da_flows:
            flow = float(da_flows[(u, v)])
        elif (v, u) in da_flows:
            flow = -float(da_flows[(v, u)])
        else:
            flow = 0.0
        congested_atc[(u, v)] = cap - flow

    return congested_atc
