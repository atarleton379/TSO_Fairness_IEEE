import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import time


from src.input import InputHandler


class Model1:
    """
    Single-hour copper-plate day-ahead market clearing
    """

    def __init__(self, hour: int, inp_hndl: InputHandler):
        """
        Run the single-hour copper-plate day-ahead market clearing optimization model for a given hour.

        Initializes the model with generator, wind, and demand data from
        ``inp_hndl``, solves the dispatch optimization, and — if an optimal
        solution is found — computes economic metrics and generates four
        output plots: merit order curve, producer profits, demand utility,
        and a combined profit/utility summary.

        Args:
            hour (int): Hour index (0-based) for which the model is solved.
            inp_hndl (InputHandler): Populated input handler exposing:
                - ``generators``   – conventional generator data
                - ``wind_farms``   – wind farm data
                - ``CF_wind``      – per-farm capacity factors
                - ``demands``      – demand data dict (must contain ``"system_load"``)

        Attributes:
            hour (int): Hour index passed at construction.
            P_gens (dict): Conventional generator data (``inp_hndl.generators``).
            P_winds (dict): Wind farm data (``inp_hndl.wind_farms``).
            CF_wind (dict): Wind capacity factors (``inp_hndl.CF_wind``).
            Demands (dict): Full demand dictionary (``inp_hndl.demands``).
            Load_sum (dict): System load (``Demands["system_load"]``).
            model (gurobipy.Model): Solved Gurobi model instance.
            out_dict (dict): Economic analysis results. Only set when
                ``model.status == GRB.OPTIMAL``; absent otherwise.
        """
        self.hour = hour
        self.P_gens = inp_hndl.generators
        self.P_winds = inp_hndl.wind_farms
        self.CF_wind = inp_hndl.CF_wind
        self.Demands = inp_hndl.demands
        self.Load_sum = inp_hndl.demands["system_load"]

        self.model = self._run_model()
        if self.model.status == gp.GRB.OPTIMAL:
            self.out_dict = self._analyze_output()
            self._plot_merit_order()
            self._plot_profit_and_utility()

    def _run_model(self) -> gp.Model:
        """
        Builds and solves the market clearing optimization model for the current hour.

        Formulates a social welfare maximization problem by maximizing the sum of
        consumer surplus (demand bids) minus total generation costs, subject to
        physical and balance constraints.

        Decision Variables:
            p_gen[i]  (continuous, >= 0): Power output for conventional generator i.
            p_wind[i] (continuous, >= 0): Power output for wind farm i.
            demand[i] (continuous, >= 0): Served demand for load i.

        Objective:
            Maximize:
                sum(demand[i] * bidding_price[i]) - sum(C[i] * p_gen[i])

        Constraints:
            - Max generation:  p_gen[i]  <= P_max[i]                         for all i in P_gens
            - Max wind output: p_wind[i] <= P_max[i] * CF_wind[hour]         for all i in P_winds
            - Max demand:      demand[i] <= load_distribution[i] * Load[hour] for all i in Demands
            - Power balance:   sum(demand) - sum(p_gen) - sum(p_wind) == 0

        Returns:
            gp.Model: The Gurobi model after optimization, regardless of solve status.
                Use model.status == GRB.OPTIMAL to check for a valid solution.
        """
        model = gp.Model("model1")
        p_gen = {}
        p_wind = {}
        demand = {}

        # Define Variables
        for i in self.P_gens:
            p_gen[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"p_gen_{i}")
        for i in self.P_winds:
            p_wind[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"p_wind_{i}")
        for i in self.Demands["load_distribution"]:
            demand[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"demand_{i}")

        # Define Objective
        model.setObjective(
            gp.quicksum(
                demand[i] * self.Demands["bidding_prices"][i]
                for i in self.Demands["bidding_prices"]
            )
            - gp.quicksum(self.P_gens[i]["C"] * p_gen[i] for i in self.P_gens),
            GRB.MAXIMIZE,
        )

        # Define Constraints
        # physical constraint conventional generators
        for i in self.P_gens:
            model.addConstr(
                p_gen[i] <= self.P_gens[i]["P_max"], name=f"max_power_gen_{i}"
            )
        # physical constraints wind farms
        for i in self.P_winds:
            model.addConstr(
                p_wind[i] <= self.P_winds[i]["P_max"] * self.CF_wind[self.hour],
                name=f"max_power_wind_{i}",
            )
        # maximum demand constraint
        for i in self.Demands["load_distribution"]:
            model.addConstr(
                demand[i]
                <= self.Demands["load_distribution"][i] * self.Load_sum[self.hour],
                name=f"max_demand_load_{i}",
            )

        # power balance constraint
        model.addConstr(
            gp.quicksum(demand[i] for i in self.Demands["load_distribution"])
            - gp.quicksum(p_gen[i] for i in self.P_gens)
            - gp.quicksum(p_wind[i] for i in self.P_winds)
            == 0,
            name="power_balance",
        )

        # Optimize the model
        solve_start_perf = time.perf_counter()
        model.optimize()
        solve_end_perf = time.perf_counter()
        model._solve_start_perf = solve_start_perf
        model._solve_end_perf = solve_end_perf
        model._solve_time_seconds = solve_end_perf - solve_start_perf

        if model.status == gp.GRB.OPTIMAL:
            print("Optimal solution found!")
        else:
            print(f"Solver status: {model.status}")
        return model

    def _analyze_output(self) -> dict:
        """
        Extracts and computes market results from the optimal solution.

        Retrieves decision variable values and dual prices from the solved
        Gurobi model and computes key economic metrics for the cleared market.

        Returns:
            dict: A dictionary containing the following market results:

                - market_clearing_price (float):
                    Dual variable (shadow price) of the power balance
                    constraint, representing the system marginal price (SMP).

                - social_welfare (float):
                    Optimal objective value; total welfare as the difference
                    between consumer value and generation cost.

                - total_operating_cost (float):
                    Sum of dispatch cost across all conventional generators:
                    sum(p_gen[i] * C[i]).

                - scheduled_power_output (list of float):
                    Dispatched power for each unit, ordered as
                    [p_gen_0, ..., p_gen_n, p_wind_0, ..., p_wind_m].

                - profit_producers (list of float):
                    Per-unit profit for each producer:
                    (MCP - C[i]) * p_gen[i] for generators,
                    MCP * p_wind[i] for wind farms (marginal costs assumed to be 0).

                - utility_demands (list of float):
                    Per-load consumer utility:
                    (bidding_price[i] - MCP) * demand[i] for each load i.
        """
        out_dict = {}
        out_dict["solve_start_perf_counter"] = getattr(self.model, "_solve_start_perf", None)
        out_dict["solve_end_perf_counter"] = getattr(self.model, "_solve_end_perf", None)
        out_dict["solve_time_seconds"] = getattr(self.model, "_solve_time_seconds", None)

        # market clearing price
        out_dict["market_clearing_price"] = self.model.getConstrByName(
            "power_balance"
        ).Pi

        # social welfare
        out_dict["social_welfare"] = self.model.ObjVal

        # total operating cost
        out_dict["total_operating_cost"] = sum(
            self.model.getVarByName(f"p_gen_{i}").X * self.P_gens[i]["C"]
            for i in self.P_gens
        )

        # profit of all producers
        out_dict["profit_producers"] = [
            self.model.getVarByName(f"p_gen_{i}").X
            * (out_dict["market_clearing_price"] - self.P_gens[i]["C"])
            for i in self.P_gens
        ] + [
            self.model.getVarByName(f"p_wind_{i}").X * out_dict["market_clearing_price"]
            for i in self.P_winds
        ]

        # utility of all demands
        out_dict["utility_demands"] = [
            self.model.getVarByName(f"demand_{i}").X
            * (self.Demands["bidding_prices"][i] - out_dict["market_clearing_price"])
            for i in self.Demands["load_distribution"]
        ]

        # scheduled power outut for merit order plotting (not part of main result)
        out_dict["scheduled power output"] = [
            self.model.getVarByName(f"p_gen_{i}").X for i in self.P_gens
        ] + [self.model.getVarByName(f"p_wind_{i}").X for i in self.P_winds]
        return out_dict

    def _plot_merit_order(self) -> None:
        """
        Plot the day-ahead market merit order curve for the current hour.

        Builds a stacked bar chart of supply offers (wind first at zero cost,
        then conventional generators sorted by ascending bid price) overlaid
        with a descending demand bid staircase. Highlights dispatched vs.
        undispatched units, marks the market clearing point, and saves the
        figure to ``results/step1_merit_order.png``.

        The plot includes:
            - **Supply bars**: wind (blue), dispatched conventional (green),
              undispatched conventional (grey).
            - **Merit order step curve**: black staircase over the supply bars.
            - **Demand bid curve**: descending staircase of consumer willingness
              to pay, sorted from highest to lowest bid price.
            - **Market clearing price (MCP)**: horizontal dashed red line.
            - **Total demand**: vertical dashed orange line.
            - **Clearing point marker**: intersection of supply and demand curves.

        Reads from:
            self.out_dict (dict): Must contain:
                - ``"market_clearing_price"`` (float): The MCP in $/MWh.
                - ``"scheduled power output"`` (list[float]): Dispatched MW per unit.
            self.P_gens (dict): Conventional generator data with ``"C"`` (bid price)
                and ``"P_max"`` (capacity) per generator.
            self.P_winds (dict): Wind farm data with ``"P_max"`` per farm.
            self.CF_wind (dict): Capacity factors keyed by hour.
            self.Demands (dict): Must contain:
                - ``"load_distribution"`` (dict): Fractional load share per demand group.
                - ``"bidding_prices"`` (dict): Willingness-to-pay per demand group.
            self.Load_sum (dict): System load keyed by hour.
            self.hour (int): Current hour index.

        Saves:
            ``results/step1_merit_order.png`` at 150 DPI.
        """
        mcp = self.out_dict["market_clearing_price"]

        # Make lists out of the generator dictionary for easier use
        gen_names = list(self.P_gens.keys())
        gen_costs = [self.P_gens[i]["C"] for i in gen_names]
        gen_caps = [self.P_gens[i]["P_max"] for i in gen_names]

        # Sort conventional generators by bid price (merit order)
        sorted_gens = sorted(zip(gen_costs, gen_caps, gen_names), key=lambda x: x[0])
        sorted_costs, sorted_caps, sorted_names = zip(*sorted_gens)

        # Wind generators (zero marginal cost, prepend to merit order)
        wind_names = list(self.P_winds.keys())
        wind_caps = [
            self.P_winds[i]["P_max"] * self.CF_wind[self.hour] for i in wind_names
        ]
        total_wind = sum(wind_caps)
        all_caps = [total_wind] + list(sorted_caps)
        all_costs = [0.0] + list(sorted_costs)
        all_labels = ["Wind"] + list(sorted_names)

        # Total demand
        total_demand = self.Load_sum[self.hour]

        # Determine which units are dispatched (scheduled output > 0), assumes wind is dispatched
        dispatched_set = set()
        scheduled = self.out_dict["scheduled power output"]
        gen_keys = list(self.P_gens.keys())
        for idx, key in enumerate(gen_keys):
            if scheduled[idx] > 1e-6:
                dispatched_set.add(key)
        if sum(scheduled[len(gen_keys) :]) > 1e-6:
            dispatched_set.add("Wind")

        # Create the Plot
        fig, ax = plt.subplots(figsize=(13, 6))

        x_cursor = 0.0
        bar_centers = []
        bar_colors = []

        color_wind = "#2196F3"
        color_dispatched = "#4CAF50"
        color_undispatched = "#BDBDBD"
        color_mcp_line = "#E53935"
        color_demand_line = "#FF6F00"

        for cap, cost, label in zip(all_caps, all_costs, all_labels):
            if cap < 1e-6:
                continue
            is_wind = label == "Wind"
            key_for_dispatch = label  # "Wind" or gen key
            dispatched = (key_for_dispatch in dispatched_set) or is_wind

            color = (
                color_wind
                if is_wind
                else (color_dispatched if dispatched else color_undispatched)
            )
            bar_centers.append(x_cursor + cap / 2)
            bar_colors.append(color)

            ax.bar(
                x_cursor + cap / 2,
                cost,
                width=cap,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                align="center",
                zorder=2,
            )
            # Label bar with generator name (rotate if narrow)
            if cap > 15:
                ax.text(
                    x_cursor + cap / 2,
                    cost + 0.15,
                    f"G{label}" if label != "Wind" else "Wind 1-6",
                    ha="center",
                    va="bottom",
                    fontsize=16,
                    rotation=0,
                    color="black",
                )
            x_cursor += cap

        total_capacity = x_cursor

        #  Step line for bid prices (merit order curve)
        step_x = []
        step_y = []
        x_cursor2 = 0.0
        for cap, cost, label in zip(all_caps, all_costs, all_labels):
            if cap < 1e-6:
                continue
            step_x.append(x_cursor2)
            step_y.append(cost)
            x_cursor2 += cap
            step_x.append(x_cursor2)
            step_y.append(cost)

        ax.step(
            step_x,
            step_y,
            where="pre",
            color="black",
            linewidth=2.5,
            linestyle="-",
            label="Bid price (merit order)",
            zorder=3,
        )

        # Market clearing price (horizontal line)
        ax.axhline(
            mcp,
            color=color_mcp_line,
            linewidth=2.5,
            linestyle="--",
            label=f"Market clearing price = ${mcp:.2f}/MWh",
            zorder=4,
        )

        # Total demand (vertical line)
        ax.axvline(
            total_demand,
            color=color_demand_line,
            linewidth=2.5,
            linestyle="--",
            label=f"Total demand = {total_demand:.1f} MW",
            zorder=4,
        )

        # Demand merit order curve
        demand_keys = list(self.Demands["load_distribution"].keys())
        demand_quantities = [
            self.Demands["load_distribution"][i] * self.Load_sum[self.hour]
            for i in demand_keys
        ]
        demand_prices = [self.Demands["bidding_prices"][i] for i in demand_keys]

        # Sort descending by bid price (highest willingness to pay first)
        sorted_demands = sorted(
            zip(demand_prices, demand_quantities, demand_keys),
            key=lambda x: x[0],
            reverse=True,
        )
        sorted_d_prices, sorted_d_quantities, _ = zip(*sorted_demands)

        # Build staircase
        demand_step_x = []
        demand_step_y = []
        x_cursor_d = 0.0
        for qty, price in zip(sorted_d_quantities, sorted_d_prices):
            if qty < 1e-6:
                continue
            demand_step_x.append(x_cursor_d)
            demand_step_y.append(price)
            x_cursor_d += qty
            demand_step_x.append(x_cursor_d)
            demand_step_y.append(price)

        ax.step(
            demand_step_x,
            demand_step_y,
            where="pre",
            color=color_demand_line,
            linewidth=2.5,
            linestyle="-",
            label="Demand bid curve",
            zorder=3,
        )

        # Intersection marker
        ax.plot(
            total_demand,
            mcp,
            marker="o",
            markersize=14,
            color=color_mcp_line,
            zorder=5,
            label=f"Clearing point ({total_demand:.1f} MW, \${mcp:.2f})",
        )

        # Axes labels & formatting
        ax.set_xlabel("Cumulative Capacity (MW)", fontsize=21)
        ax.set_ylabel("Bid Price ($/MWh)", fontsize=21)
        ax.set_xlim(0, max(total_capacity, x_cursor_d) * 1.02)
        ax.set_ylim(-0.5, max(all_costs) * 2 + 1)
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7, zorder=1)
        ax.tick_params(axis="both", labelsize=15)

        # Legend
        legend_patches = [
            mpatches.Patch(color=color_wind, label="Wind (zero marginal cost)"),
            mpatches.Patch(color=color_dispatched, label="Dispatched (conventional)"),
            mpatches.Patch(color=color_undispatched, label="Not dispatched"),
        ]
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(
            handles=legend_patches + handles,
            fontsize=12,
            loc="upper right",
            framealpha=0.9,
        )

        plt.tight_layout()
        plt.savefig("results/step1_merit_order.png", dpi=150)

    def _plot_profit_and_utility(self):
        """
        Plot producer profits and consumer utilities side-by-side for the current hour.

        Produces a two-panel figure:
            - **Left panel**: Bar chart of profit ($/hour) for each producer —
              conventional generators (green, labelled G1, G2, …) followed by
              wind farms (blue, labelled W1, W2, …). Dollar values are annotated
              above each non-zero bar.
            - **Right panel**: Bar chart of utility ($/hour) for each demand group
              (orange, labelled D1, D2, …), also annotated above non-zero bars.

        Reads from:
            self.out_dict (dict): Must contain:
                - ``"profit_producers"`` (list[float]): Profit per producer,
                  ordered as conventional generators first, then wind farms.
                - ``"utility_demands"`` (list[float]): Utility per demand group.

        Saves:
            ``results/step1_profit_utility.png`` at 150 DPI.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        # --- Left panel: Producer Profits ---
        x_prod = range(len(self.out_dict["profit_producers"]))
        colors = ["#2196F3" if i >= len(x_prod) - 6 else "#4CAF50" for i in x_prod]
        labels_prod = [
            f"W{i - len(x_prod) + 7}" if i >= len(x_prod) - 6 else f"G{i+1}"
            for i in x_prod
        ]

        bars1 = ax1.bar(
            x_prod,
            self.out_dict["profit_producers"],
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            width=0.6,
        )

        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 20,
                    f"${height:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color="#333333",
                )

        ax1.set_xlabel("Producer", fontsize=21, labelpad=10)
        ax1.set_ylabel("Profit ($)", fontsize=21, labelpad=10)
        ax1.set_xticks(x_prod)
        ax1.set_xticklabels(labels_prod, fontsize=14)
        ax1.tick_params(axis="both", labelsize=13)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.yaxis.grid(True, linestyle="--", alpha=0.5, color="grey")
        ax1.set_axisbelow(True)

        # colour legend for producers
        ax1.legend(
            handles=[
                Patch(facecolor="#4CAF50", label="Conventional generators"),
                Patch(facecolor="#2196F3", label="Wind farms"),
            ],
            fontsize=15,
            framealpha=0.9,
            loc="upper left",
        )

        # --- Right panel: Consumer Utility ---
        x_dem = range(len(self.out_dict["utility_demands"]))

        bars2 = ax2.bar(
            x_dem,
            self.out_dict["utility_demands"],
            color="#FF6F00",
            edgecolor="white",
            linewidth=0.8,
            width=0.6,
        )

        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 20,
                    f"${height:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color="#333333",
                )

        ax2.set_xlabel("Demand", fontsize=21, labelpad=10)
        ax2.set_ylabel("Utility ($)", fontsize=21, labelpad=10)
        ax2.set_xticks(x_dem)
        ax2.set_xticklabels([f"D{i+1}" for i in x_dem], fontsize=14)
        ax2.tick_params(axis="both", labelsize=13)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.yaxis.grid(True, linestyle="--", alpha=0.5, color="grey")
        ax2.set_axisbelow(True)

        ax2.legend(
            handles=[
                Patch(facecolor="#FF6F00", label="Demand"),
            ],
            fontsize=15,
            framealpha=0.9,
            loc="upper left",
        )
        plt.tight_layout()
        plt.savefig("results/step1_profit_utility.png", dpi=150, bbox_inches="tight")
