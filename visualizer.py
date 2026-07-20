import json
import os
import matplotlib

# Use a non-interactive backend: this module only ever saves figures to disk
# (or shows them via plt.show() when run standalone), and the interactive Qt
# backend can fail to load on some machines (missing platform plugin).
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np

from input import InputHandler
from model_helpers import reference_incident_by_zone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zone_palette(zones):
    """Return a color per zone."""
    cmap = cm.get_cmap("tab20", max(len(zones), 1))
    return {z: cmap(i) for i, z in enumerate(sorted(zones.keys()))}


def _zone_positions(zones):
    """Create a simple zone-level layout (not tied to the 24-bus topology)."""
    zone_ids = sorted(zones.keys())
    n = len(zone_ids)
    pos = {}
    for i, z in enumerate(zone_ids):
        angle = 2 * np.pi * i / n
        pos[z] = (1.4 * np.cos(angle), 1.2 * np.sin(angle))
    return pos


def _normalize_flow_keys(raw: dict) -> dict:
    """Convert any key format ((1,2), [1,2], '1, 2', '[1, 2]') to (int, int) tuples."""
    out = {}
    for k, v in raw.items():
        if isinstance(k, tuple):
            out[(int(k[0]), int(k[1]))] = v
        elif isinstance(k, list):
            out[(int(k[0]), int(k[1]))] = v
        elif isinstance(k, str):
            parts = k.strip("[]()").split(",")
            if len(parts) >= 2:
                out[(int(parts[0].strip()), int(parts[1].strip()))] = v
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def _normalize_zone_dict(raw: dict) -> dict:
    """Ensure zone-keyed dict has integer keys."""
    return {int(k): v for k, v in raw.items()}


def _pair_key(u, v):
    return (min(int(u), int(v)), max(int(u), int(v)))


def _pair_has_capacity(a, b, atc, congested_atc, tol=1e-9):
    """True if either direction of the zone pair has nonzero ATC, baseline or congested."""
    values = (
        float(atc.get((a, b), 0.0)),
        float(atc.get((b, a), 0.0)),
        float(congested_atc.get((a, b), 0.0)),
        float(congested_atc.get((b, a), 0.0)),
    )
    return any(abs(v) > tol for v in values)


def _text_box_size(text: str, char_w=0.0125, line_h=0.040, pad=0.018):
    """Approximate text box size in data units for collision checks."""
    lines = text.split("\n")
    max_line = max((len(line) for line in lines), default=0)
    width = char_w * max_line + pad
    height = line_h * len(lines) + pad
    return width, height


def _boxes_overlap(box_a, box_b):
    """Axis-aligned rectangle intersection test for label collision checks."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _place_non_overlapping_label(midpoint, perp, tangent, text, used_boxes,
                                 base_offset=0.10, step=0.055, max_tries=40):
    """Pick a label position by walking away from midpoint until no collision."""
    mx, my = midpoint
    px, py = perp
    tx, ty = tangent
    w, h = _text_box_size(text)

    for k in range(max_tries):
        band = k // 4
        variant = k % 4

        # Try both sides of the normal first, then repeat with a small tangent
        # shift to escape local clusters around the same midpoint.
        normal_sign = 1 if variant in (0, 2) else -1
        tangent_sign = 0 if variant < 2 else (1 if variant == 2 else -1)

        offset = base_offset + step * band
        tangent_shift = 0.035 * band if tangent_sign != 0 else 0.0

        lx = mx + normal_sign * offset * px + tangent_sign * tangent_shift * tx
        ly = my + normal_sign * offset * py + tangent_sign * tangent_shift * ty
        box = (lx - 0.5 * w, ly - 0.5 * h, lx + 0.5 * w, ly + 0.5 * h)

        if not any(_boxes_overlap(box, old_box) for old_box in used_boxes):
            used_boxes.append(box)
            return lx, ly

    # Fallback: keep the final candidate even if overlap remains.
    used_boxes.append(box)
    return lx, ly


def _draw_edge_flow_labels(G, pos, ax, edge_labels, start_offset=0.08):
    """Draw edge labels at slightly staggered manual positions to avoid overlap."""
    used_boxes = []
    for i, (u, v) in enumerate(G.edges()):
        if (u, v) not in edge_labels:
            continue

        x1, y1 = pos[u]
        x2, y2 = pos[v]
        mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

        dx, dy = x2 - x1, y2 - y1
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        px, py = -dy / length, dx / length

        dynamic_offset = start_offset + 0.04 / max(length, 0.15)
        lx, ly = _place_non_overlapping_label(
            (mx, my),
            (px, py),
            (dx / length, dy / length),
            edge_labels[(u, v)],
            used_boxes,
            base_offset=dynamic_offset,
            step=0.055,
            max_tries=40,
        )

        ax.text(
            lx,
            ly,
            edge_labels[(u, v)],
            fontsize=7,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.2),
        )


def _draw_pair_flow_labels(pos, ax, pair_labels, start_offset=0.10):
    """Draw one consolidated label per undirected zone pair."""
    pair_keys = sorted(pair_labels.keys())
    used_boxes = []

    for i, (a, b) in enumerate(pair_keys):
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

        dx, dy = x2 - x1, y2 - y1
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        px, py = -dy / length, dx / length

        dynamic_offset = start_offset + 0.05 / max(length, 0.15)
        lx, ly = _place_non_overlapping_label(
            (mx, my),
            (px, py),
            (dx / length, dy / length),
            pair_labels[(a, b)],
            used_boxes,
            base_offset=dynamic_offset,
            step=0.060,
            max_tries=44,
        )

        ax.text(
            lx,
            ly,
            pair_labels[(a, b)],
            fontsize=7,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.2),
        )


# ---------------------------------------------------------------------------
# Panel 1 – ATC congestion by day-ahead model
# ---------------------------------------------------------------------------

def plot_atc_congestion(inp_hndl, da_results, atc_results, congested_results, ax):
    zones = inp_hndl.zones
    zone_pos = _zone_positions(zones)

    atc = _normalize_flow_keys(atc_results or {})
    congested_atc = _normalize_flow_keys(congested_results or {})

    # If no congested ATC is provided, fall back to baseline ATC.
    if not congested_atc:
        congested_atc = dict(atc)

    # Mode A (IEEE pipeline): color by DA-induced delta from baseline ATC.
    # Mode B (toy/manual directional): if no DA delta exists, color by deviation
    # from pair baseline (mean of both directions) to still show directional skew.
    has_da_delta = any(
        abs(float(congested_atc.get((u, v), 0.0)) - float(atc.get((u, v), 0.0))) > 1e-6
        for (u, v) in set(atc.keys()) | set(congested_atc.keys())
    )

    G = nx.DiGraph()
    G.add_nodes_from(zones.keys())

    # Add directional ATC edges for all zone pairs seen in baseline or congested
    # ATC, skipping pairs with zero capacity in both directions (no real corridor).
    all_edges = set(atc.keys()) | set(congested_atc.keys())
    for (u, v) in all_edges:
        if int(u) == int(v):
            continue
        if int(u) not in zones or int(v) not in zones:
            continue
        a, b = _pair_key(u, v)
        if not _pair_has_capacity(a, b, atc, congested_atc):
            continue
        G.add_edge(int(u), int(v))

    # Draw nodes first with a single shared color so the edges carry the signal.
    shared_node_color = "lightsteelblue"
    nx.draw_networkx_nodes(G, zone_pos, ax=ax, nodelist=sorted(zones.keys()), node_color=shared_node_color,
                           node_size=900, edgecolors="black", linewidths=0.8)
    nx.draw_networkx_labels(G, zone_pos, ax=ax, labels={z: f"Z{z}" for z in sorted(zones.keys())},
                            font_size=8, font_weight="bold")

    # Directional edge styling: show the original baseline ATC as a faint dashed
    # reference, then draw the adjusted ATC on top with congestion-based coloring.
    edge_colors = []
    edge_widths = []
    baseline_edges = []
    adjusted_edges = []
    for u, v, data in G.edges(data=True):
        base_cap = float(atc.get((u, v), congested_atc.get((u, v), 0.0)))
        this_dir_cap = float(congested_atc.get((u, v), base_cap))

        if has_da_delta:
            delta = this_dir_cap - base_cap
        else:
            opposite_base = float(atc.get((v, u), base_cap))
            pair_base = 0.5 * (base_cap + opposite_base)
            delta = this_dir_cap - pair_base

        baseline_edges.append((u, v))
        adjusted_edges.append((u, v))
        if abs(delta) <= 1e-6:
            edge_color = "lightgray"
            edge_width = 1.0
        elif delta < 0:
            edge_color = "red"
            edge_width = 1.2 + 3.0 * abs(delta) / max(base_cap, 1.0)
        else:
            edge_color = "green"
            edge_width = 1.2 + 3.0 * abs(delta) / max(base_cap, 1.0)

        edge_colors.append(edge_color)
        edge_widths.append(edge_width)

    nx.draw_networkx_edges(G, zone_pos, ax=ax, edgelist=baseline_edges, edge_color="lightgray",
                           width=1.0, style="dashed", arrows=True, arrowstyle="-|>", arrowsize=10,
                           alpha=0.35, connectionstyle="arc3,rad=0.15")
    nx.draw_networkx_edges(G, zone_pos, ax=ax, edgelist=adjusted_edges, edge_color=edge_colors,
                           width=edge_widths, arrows=True, arrowstyle="-|>", arrowsize=12, alpha=0.85,
                           connectionstyle="arc3,rad=0.15")

    pair_labels = {}
    for a, b in sorted({_pair_key(u, v) for u, v in G.edges()}):
        base_ab = float(atc.get((a, b), 0.0))
        base_ba = float(atc.get((b, a), 0.0))
        pair_base = 0.5 * (base_ab + base_ba)
        adj_ab = float(congested_atc.get((a, b), base_ab))
        adj_ba = float(congested_atc.get((b, a), base_ba))
        pair_labels[(a, b)] = (
            f"line_cap = {pair_base:.1f}\n"
            f"cong_cap {a},{b}={adj_ab:.1f}\n"
            f"cong_cap {b},{a}={adj_ba:.1f}"
        )
    _draw_pair_flow_labels(zone_pos, ax, pair_labels)

    ax.set_title("ATC by direction\n(red = reduced flow potential, green = increased flow potential)", fontsize=10, fontweight="bold")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Panel 2 – reserve responsibility and sharing flows
# ---------------------------------------------------------------------------

def plot_reserves(inp_hndl, reserve_results, ax, label="Reserve"):
    zones = inp_hndl.zones
    zone_pos = _zone_positions(zones)
    ri_by_zone = _normalize_zone_dict(reference_incident_by_zone(inp_hndl))

    reserve = _normalize_zone_dict(reserve_results.get("reserve", {}))
    sharing_flows = _normalize_flow_keys(reserve_results.get("sharing_flows", {}))

    G = nx.DiGraph()
    G.add_nodes_from(zones.keys())
    tol = 1e-3
    for (u, v), flow in sharing_flows.items():
        if flow > tol:
            G.add_edge(int(u), int(v), flow=float(flow))

    max_reserve = max(reserve.values(), default=1)
    node_sizes = [300 + 900 * reserve.get(z, 0) / max_reserve for z in sorted(zones.keys())]
    shared_node_color = "lightsteelblue"

    nx.draw_networkx_nodes(G, zone_pos, ax=ax, nodelist=sorted(zones.keys()),
                           node_color=shared_node_color, node_size=node_sizes, edgecolors="black")

    for z in sorted(zones.keys()):
        x, y = zone_pos[z]
        ax.text(x, y + 0.22, f"RI={ri_by_zone.get(z, 0):.0f}", fontsize=7, ha="center", va="bottom")
        ax.text(x, y - 0.02, f"R={reserve.get(z, 0):.0f}", fontsize=8, ha="center", va="center",
                fontweight="bold")
        ax.text(x, y - 0.22, f"Z{z}", fontsize=8, ha="center", va="top")

    if G.edges():
        flows = [G[u][v]["flow"] for u, v in G.edges()]
        max_flow = max(flows) if flows else 1.0
        nx.draw_networkx_edges(G, zone_pos, ax=ax, width=[1 + 4 * f / max_flow for f in flows],
                               edge_color="darkorange", arrows=True, arrowsize=16,
                               connectionstyle="arc3,rad=0.15")

        pair_labels = {}
        for a, b in sorted({_pair_key(u, v) for u, v in G.edges()}):
            ab = float(G[a][b]["flow"]) if G.has_edge(a, b) else 0.0
            ba = float(G[b][a]["flow"]) if G.has_edge(b, a) else 0.0
            pair_labels[(a, b)] = (
                f"flow {a},{b}={ab:.0f}\n"
                f"flow {b},{a}={ba:.0f}"
            )
        _draw_pair_flow_labels(zone_pos, ax, pair_labels, start_offset=0.10)
    tot_r = reserve_results["total_procurement"]
    ax.set_title(f"{label}\n(node text = reserve procured; text above = reference incident)\nreserve cost = {tot_r}" , fontsize=10, fontweight="bold")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def _find_reserve_variant(results: dict, preferred_keys: list[str], label: str):
    for key in preferred_keys:
        if results.get(key):
            return results[key], label
    return None, None


def visualize(inp_hndl, results: dict, output_file: str | None = None):
    """Generate a two-panel figure: ATC congestion, then reserve responsibility."""
    reserve_result, reserve_label = _find_reserve_variant(
        results,
        ["mmf_standard_reserve", "mmf standard reserve"],
        "MMF Reserve",
    )
    if not reserve_result:
        reserve_result, reserve_label = _find_reserve_variant(
            results,
            ["alpha_fair_reserve", "alpha fair reserve"],
            "Alpha-Fair Reserve",
        )
    if not reserve_result:
        reserve_result, reserve_label = _find_reserve_variant(
            results,
            ["pf_standard_reserve", "pf standard reserve"],
            "PF Reserve",
        )
    if not reserve_result:
        reserve_result, reserve_label = _find_reserve_variant(
            results,
            ["standard_reserve", "standard reserve"],
            "Standard Reserve",
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    plot_atc_congestion(inp_hndl, results.get("day_ahead_zonal", {}), results.get("atc", {}),
                        results.get("congested_atc", {}), axes[0])
    if reserve_result and reserve_label:
        plot_reserves(inp_hndl, reserve_result, axes[1], label=reserve_label)
    else:
        axes[1].text(0.5, 0.5, "No reserve results available", ha="center", va="center")
        axes[1].axis("off")

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {output_file}")
    else:
        plt.show()


def visualize_reserve_variant(inp_hndl, results: dict, reserve_key: str, reserve_label: str, output_file: str | None = None):
    """Create a day-ahead + reserve comparison figure for one reserve model."""
    reserve_result = results.get(reserve_key)
    if reserve_result is None:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    plot_atc_congestion(inp_hndl, results.get("day_ahead_zonal", {}), results.get("atc", {}),
                        results.get("congested_atc", {}), axes[0])
    plot_reserves(inp_hndl, reserve_result, axes[1], label=reserve_label)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {output_file}")
    else:
        plt.show()

    return output_file


def visualize_all_reserve_models(inp_hndl, results: dict, output_dir: str | None = None):
    """Save one figure per reserve model plus a combined figure with all panels."""
    if output_dir is None:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    variants = [
        ("standard_reserve", "Standard Reserve"),
        ("standard reserve", "Standard Reserve"),
        ("pf_standard_reserve", "PF Reserve"),
        ("mmf_standard_reserve", "MMF Reserve"),
        ("alpha_fair_reserve", "Alpha-Fair Reserve"),
    ]

    saved_paths = []
    for key, label in variants:
        if results.get(key):
            slug = key.replace(" ", "_")
            out_path = os.path.join(output_dir, f"{slug}_comparison.png")
            visualize_reserve_variant(inp_hndl, results, key, label, output_file=out_path)
            saved_paths.append(out_path)

    reserve_panels = []
    for key, label in [
        ("standard_reserve", "Standard Reserve"),
        ("pf_standard_reserve", "PF Reserve"),
        ("mmf_standard_reserve", "MMF Reserve"),
        ("alpha_fair_reserve", "Alpha-Fair Reserve"),
    ]:
        reserve_result = results.get(key) or results.get(key.replace("_", " "))
        if reserve_result:
            reserve_panels.append((reserve_result, label))

    # Grid sizing is dynamic: 1 ATC panel plus one panel per available reserve
    # model, laid out with up to 3 columns and as many rows as needed.
    total_panels = 1 + len(reserve_panels)
    n_cols = min(3, total_panels)
    n_rows = -(-total_panels // n_cols)  # ceil division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), squeeze=False)
    flat_axes = [axes[r][c] for r in range(n_rows) for c in range(n_cols)]

    plot_atc_congestion(inp_hndl, results.get("day_ahead_zonal", {}), results.get("atc", {}),
                        results.get("congested_atc", {}), flat_axes[0])

    if reserve_panels:
        for ax, (reserve_result, label) in zip(flat_axes[1:], reserve_panels):
            plot_reserves(inp_hndl, reserve_result, ax, label=label)
    else:
        flat_axes[1].text(0.5, 0.5, "No reserve model results available", ha="center", va="center")
        flat_axes[1].axis("off")

    # Turn off any unused trailing axes (grid may have more slots than panels).
    for ax in flat_axes[1 + len(reserve_panels):]:
        ax.axis("off")

    fig.suptitle("Day-ahead ATC and reserve procurement results", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    combined_path = os.path.join(output_dir, "all_reserve_models.png")
    plt.savefig(combined_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {combined_path}")
    saved_paths.append(combined_path)
    return saved_paths


def visualize_from_folder(results_folder: str, inp_hndl=None, output_file: str | None = None):
    """Load JSON results from a run folder and visualize."""
    if inp_hndl is None:
        inp_hndl = InputHandler()

    results = {}
    for fname in os.listdir(results_folder):
        if fname.endswith(".json"):
            with open(os.path.join(results_folder, fname)) as f:
                results[fname[:-5]] = json.load(f)

    da = results.get("day_ahead_zonal", {})
    if "cross_zonal_flows" in da:
        da["cross_zonal_flows"] = _normalize_flow_keys(da["cross_zonal_flows"])

    if output_file:
        visualize(inp_hndl, results, output_file=output_file)
    else:
        visualize_all_reserve_models(inp_hndl, results, output_dir=results_folder)


if __name__ == "__main__":
    inp_hndl = InputHandler()
    results_base = "code/results"
    if os.path.exists(results_base):
        folders = sorted(os.listdir(results_base))
        if folders:
            latest = os.path.join(results_base, folders[-1])
            print(f"Visualizing results from: {latest}")
            visualize_from_folder(latest, inp_hndl)
        else:
            print("No result folders found. Run main.py first.")
    else:
        print(f"Results directory '{results_base}' not found. Run main.py first.")
