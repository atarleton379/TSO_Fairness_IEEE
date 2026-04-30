import json
import os
import matplotlib.pyplot as plt

SENS_FOLDER = os.path.join(os.path.dirname(__file__), "sensitivity_results", "sensitivity")
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "results")

# Load all sensitivity result files
sens_data = {}
for fname in os.listdir(SENS_FOLDER):
    if fname.startswith("model3_sens_") and fname.endswith(".json"):
        factor = float(fname.replace("model3_sens_", "").replace(".json", ""))
        with open(os.path.join(SENS_FOLDER, fname)) as f:
            sens_data[factor] = json.load(f)

# def find_latest_model2_result(results_folder):
#     """Return the newest model2.json found in timestamped result folders."""
#     if not os.path.isdir(results_folder):
#         return None

#     candidate_dirs = []
#     for entry in os.listdir(results_folder):
#         full_path = os.path.join(results_folder, entry)
#         if os.path.isdir(full_path) and entry != "sensitivity":
#             model2_path = os.path.join(full_path, "model2.json")
#             if os.path.isfile(model2_path):
#                 candidate_dirs.append(entry)

#     if not candidate_dirs:
#         return None

#     latest_dir = sorted(candidate_dirs)[-1]
#     return os.path.join(results_folder, latest_dir, "model2.json")


# def plot_model2_soc_and_price(model2_json_path, out_path=None):
#     """Plot battery SOC and market clearing price over periods for Model 2."""
#     with open(model2_json_path) as f:
#         out_dict = json.load(f)

#     soc = out_dict.get("battery_soc")
#     mcp = out_dict.get("market_clearing_price")
#     if not soc or not mcp:
#         print(f"Skipping Model 2 SOC plot: missing battery_soc or market_clearing_price in {model2_json_path}")
#         return

#     periods = list(range(1, len(soc) + 1))
#     signed_volume = out_dict.get("battery_signed_volume")

#     fig, ax_soc = plt.subplots(figsize=(10, 5))
#     ax_soc.plot(periods, soc, color="teal", marker="o", linewidth=2, label="Battery SOC")
#     ax_soc.set_xlabel("Period", fontsize=12)
#     ax_soc.set_ylabel("Battery SOC (MWh)", color="teal", fontsize=12)
#     ax_soc.tick_params(axis="y", labelcolor="teal")
#     ax_soc.set_xticks(periods)
#     ax_soc.grid(axis="both", linestyle="--", alpha=0.25)

#     if signed_volume and len(signed_volume) == len(periods):
#         ax_soc.bar(periods, signed_volume, color="gray", alpha=0.2, width=0.7, label="Battery signed volume (+charge, -discharge)")

#     ax_mcp = ax_soc.twinx()
#     ax_mcp.plot(periods, mcp, color="darkred", marker="s", linewidth=2, label="Market clearing price")
#     ax_mcp.set_ylabel("Market clearing price (EUR/MWh)", color="darkred", fontsize=12)
#     ax_mcp.tick_params(axis="y", labelcolor="darkred")

#     lines_soc, labels_soc = ax_soc.get_legend_handles_labels()
#     lines_mcp, labels_mcp = ax_mcp.get_legend_handles_labels()
#     ax_soc.legend(lines_soc + lines_mcp, labels_soc + labels_mcp, loc="upper left", fontsize=9)
#     plt.title("Model 2: SOC and Market Clearing Price Across Periods", fontsize=13)
#     plt.tight_layout()

#     if out_path is None:
#         out_path = os.path.join(os.path.dirname(model2_json_path), "model2_soc_and_mcp.png")

#     plt.savefig(out_path, dpi=150)
#     plt.show()
#     print(f"Chart saved to: {out_path}")


# Compute metrics for each sensitivity factor
spread_results = {}
variance_results = {}
congestion_rent_results = {}
power_consumed_results = {}
for factor, out_dict in sens_data.items():
    nodal_prices = [-v for k, v in out_dict.items() if k.startswith("node_") and k.endswith("_price")]
    spread_results[factor] = max(nodal_prices) - min(nodal_prices)
    mean = sum(nodal_prices) / len(nodal_prices)
    variance_results[factor] = sum((p - mean) ** 2 for p in nodal_prices) / len(nodal_prices)
    congestion_rent_results[factor] = out_dict.get("total_congestion_rent", 0)
    power_consumed_results[factor] = out_dict.get("total_power_consumed", 0)

# Sort by sensitivity factor
sorted_factors = sorted(spread_results.keys())
x_labels = [f"{int(f*100)}%" for f in sorted_factors]
spread_values = [spread_results[f] for f in sorted_factors]
variance_values = [variance_results[f] for f in sorted_factors]
congestion_rent_values = [congestion_rent_results[f] for f in sorted_factors]
power_consumed_values = [power_consumed_results[f] for f in sorted_factors]

def plot_bar(x_labels, y_values, ylabel, title, out_path, color):
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x_labels, y_values, color=color, edgecolor="black")
    ax.set_xlabel("Line Capacity Available (%)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_ylim(0, max(y_values) * 1.15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Chart saved to: {out_path}")

if sens_data:
    plot_bar(
        x_labels, spread_values,
        ylabel="Max Nodal Price Difference (€/MWh)",
        title="Sensitivity Analysis: Line Capacity vs Max Nodal Price Spread",
        out_path=os.path.join(SENS_FOLDER, "nodal_price_spread.png"),
        color="steelblue"
    )

    plot_bar(
        x_labels, variance_values,
        ylabel="Nodal Price Variance (€/MWh)²",
        title="Sensitivity Analysis: Line Capacity vs Nodal Price Variance",
        out_path=os.path.join(SENS_FOLDER, "nodal_price_variance.png"),
        color="darkorange"
    )

    plot_bar(
        x_labels, congestion_rent_values,
        ylabel="Total Congestion Rent (€)",
        title="Sensitivity Analysis: Line Capacity vs Total Congestion Rent",
        out_path=os.path.join(SENS_FOLDER, "congestion_rent.png"),
        color="firebrick"
    )

    plot_bar(
        x_labels, power_consumed_values,
        ylabel="Total Power Consumed (MWh)",
        title="Sensitivity Analysis: Line Capacity vs Total Power Consumed",
        out_path=os.path.join(SENS_FOLDER, "total_power_consumed.png"),
        color="seagreen"
    )
else:
    print(f"No model3 sensitivity files found in: {SENS_FOLDER}")

def plot_nodal_price_pockets(sens_data, sorted_factors, out_path):
    """
    Line plot with one line per node showing nodal price vs transmission capacity.
    Reveals price pockets forming as capacity is restricted.
    """
    # Gather all node keys from the first result
    sample = next(iter(sens_data.values()))
    nodes = sorted([k for k in sample if k.startswith("node_") and k.endswith("_price")])

    x_vals = [int(f * 100) for f in sorted_factors]  # numeric x for proper ordering

    fig, ax = plt.subplots(figsize=(12, 6))
    for node_key in nodes:
        node_label = node_key.replace("node_", "node ").replace("_price", "")
        prices = [-sens_data[f][node_key] for f in sorted_factors]
        ax.plot(x_vals, prices, marker="o", linewidth=1.2, markersize=4, label=node_label, alpha=0.8)

    ax.set_xlabel("Line Capacity Available (%)", fontsize=12)
    ax.set_ylabel("Nodal Price (€/MWh)", fontsize=12)
    ax.set_title("Nodal Price Pockets vs Transmission Capacity", fontsize=13)
    ax.set_xticks(x_vals)
    ax.set_xticklabels([f"{x}%" for x in x_vals])
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Chart saved to: {out_path}")

if sens_data:
    plot_nodal_price_pockets(
        sens_data, sorted_factors,
        out_path=os.path.join(SENS_FOLDER, "nodal_price_pockets.png")
    )

# ── Model 4 (Zonal) sensitivity visualizations ────────────────────────────────

sens_data_m4 = {}
for fname in os.listdir(SENS_FOLDER):
    if fname.startswith("model4_sens_") and fname.endswith(".json"):
        factor = float(fname.replace("model4_sens_", "").replace(".json", ""))
        with open(os.path.join(SENS_FOLDER, fname)) as f:
            sens_data_m4[factor] = json.load(f)

if sens_data_m4:
    m4_spread = {}
    m4_variance = {}
    m4_power_consumed = {}
    for factor, out_dict in sens_data_m4.items():
        zone_prices = [-v for k, v in out_dict.items() if k.startswith("zone_") and k.endswith("_price")]
        m4_spread[factor] = max(zone_prices) - min(zone_prices)
        mean = sum(zone_prices) / len(zone_prices)
        m4_variance[factor] = sum((p - mean) ** 2 for p in zone_prices) / len(zone_prices)
        m4_power_consumed[factor] = out_dict.get("total_power_consumed", 0)

    m4_sorted_factors = sorted(m4_spread.keys())
    m4_x_labels = [f"{int(f*100)}%" for f in m4_sorted_factors]

    M4_FOLDER = os.path.join(SENS_FOLDER, "model4")
    os.makedirs(M4_FOLDER, exist_ok=True)

    plot_bar(
        m4_x_labels, [m4_spread[f] for f in m4_sorted_factors],
        ylabel="Max Zonal Price Difference (€/MWh)",
        title="Model 4 Sensitivity: Line Capacity vs Max Zonal Price Spread",
        out_path=os.path.join(M4_FOLDER, "zonal_price_spread.png"),
        color="steelblue"
    )

    plot_bar(
        m4_x_labels, [m4_variance[f] for f in m4_sorted_factors],
        ylabel="Zonal Price Variance (€/MWh)²",
        title="Model 4 Sensitivity: Line Capacity vs Zonal Price Variance",
        out_path=os.path.join(M4_FOLDER, "zonal_price_variance.png"),
        color="darkorange"
    )

    plot_bar(
        m4_x_labels, [m4_power_consumed[f] for f in m4_sorted_factors],
        ylabel="Total Power Consumed (MWh)",
        title="Model 4 Sensitivity: Line Capacity vs Total Power Consumed",
        out_path=os.path.join(M4_FOLDER, "total_power_consumed.png"),
        color="seagreen"
    )

    # Zonal price pockets (equivalent of nodal price pockets for 3 zones)
    sample_m4 = next(iter(sens_data_m4.values()))
    zone_keys = sorted([k for k in sample_m4 if k.startswith("zone_") and k.endswith("_price")])
    m4_x_vals = [int(f * 100) for f in m4_sorted_factors]

    fig, ax = plt.subplots(figsize=(10, 5))
    for zk in zone_keys:
        label = zk.replace("zone_", "Zone ").replace("_price", "")
        prices = [-sens_data_m4[f][zk] for f in m4_sorted_factors]
        ax.plot(m4_x_vals, prices, marker="o", linewidth=2, markersize=6, label=label)

    ax.set_xlabel("Line Capacity Available (%)", fontsize=12)
    ax.set_ylabel("Zonal Price (€/MWh)", fontsize=12)
    ax.set_title("Model 4: Zonal Price Divergence vs Transmission Capacity", fontsize=13)
    ax.set_xticks(m4_x_vals)
    ax.set_xticklabels([f"{x}%" for x in m4_x_vals])
    ax.legend(fontsize=10)
    plt.tight_layout()
    zonal_pockets_path = os.path.join(M4_FOLDER, "zonal_price_pockets.png")
    plt.savefig(zonal_pockets_path, dpi=150)
    plt.show()
    print(f"Chart saved to: {zonal_pockets_path}")
else:
    print(f"No model4 sensitivity files found in: {SENS_FOLDER}")

latest_model2_file = find_latest_model2_result(RESULTS_FOLDER)
if latest_model2_file:
    plot_model2_soc_and_price(latest_model2_file)
else:
    print(f"No model2.json found under: {RESULTS_FOLDER}")
