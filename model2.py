import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
import time

from src.input import InputHandler


class Model2:
    """
    Multiple hours copper-plate day-ahead market clearing with battery storage
    """

    def __init__(self, hours: int, inp_hndl: InputHandler, name="step2"):
        """
        Initializes and runs the optimization model for 24 hours. If optimal solution is found, computes key economic metrics.

        Args:
            hour (int): The hour of the day for which the model is run.
            inp_hndl (InputHandler): An InputHandler instance containing all
                necessary input data, including generators, wind farms,
                capacity factors, and demand information.

        Attributes:
            hours (int): Number of hours considered in optimization problem.
            P_gens (dict): Conventional generator data from the input handler.
            P_winds (dict): Wind farm data from the input handler.
            CF_wind (dict): Wind capacity factors from the input handler.
            Demands (dict): Full demand data dictionary from the input handler.
            Load_sum (dict): System load extracted from the demands dictionary.
            P_ch_max_bat (int): Maximum charging power of battery (in kW).
            P_dis_max_bat (int): Maximum discharging power of battery (in kW).
            E_cap_bat (int): Capacity of battery (in kWh).
            Effic_ch (float): Charging efficiency of battery.
            Effic_dis (float): Discharging efficiency of battery.
            model (gurobipy.Model): The Gurobi model after optimization.
            out_dict (dict): Output analysis results, set only if the model
                reaches an optimal solution.
        """
        self.hours = hours
        self.P_gens = inp_hndl.generators
        self.P_winds = inp_hndl.wind_farms
        self.CF_wind = inp_hndl.CF_wind
        self.Demands = inp_hndl.demands
        self.Load_sum = inp_hndl.demands["system_load"]
        self.P_ch_max_bat = inp_hndl.P_ch_max_bat
        self.P_dis_max_bat = inp_hndl.P_dis_max_bat
        self.E_cap_bat = inp_hndl.E_cap_bat
        self.Effic_ch = inp_hndl.Effic_ch
        self.Effic_dis = inp_hndl.Effic_dis

        self.model = self._run_model()
        if self.model.status == gp.GRB.OPTIMAL:
            self.out_dict = self._analyze_output()

    def _run_model(self) -> gp.Model:
        """
        Builds and solves the multi-period market clearing optimization model
        with battery storage for one day.

        Extends the single-hour formulation by co-optimizing dispatch and
        battery charge/discharge across all hours, capturing inter-temporal
        arbitrage opportunities from storage.

        Decision Variables (indexed by hour t in [1, hours]):
            p_gen[i,t]    (continuous, >= 0): Output of conventional generator i at hour t.
            p_wind[i,t]   (continuous, >= 0): Output of wind farm i at hour t.
            demand[i,t]   (continuous, >= 0): Served demand for load i at hour t.
            p_ch_bat[t]   (continuous, >= 0): Battery charging power at hour t.
            p_dis_bat[t]  (continuous, >= 0): Battery discharging power at hour t.
            e_bat[t]      (continuous, >= 0): Stored energy in battery at end of hour t.

        Objective:
            Maximize total social welfare across all hours:
                sum_t [ sum_i(demand[i,t] * bidding_price[i])
                      - sum_i(C[i] * p_gen[i,t]) ]

        Constraints (for each hour t):
            - Max generation:    p_gen[i,t]  <= P_max[i]                          for all i in P_gens
            - Max wind output:   p_wind[i,t] <= P_max[i] * CF_wind[t]             for all i in P_winds
            - Max demand:        demand[i,t] <= load_distribution[i] * Load[t]    for all i in Demands
            - Max charge:        p_ch_bat[t]  <= P_ch_max
            - Max discharge:     p_dis_bat[t] <= P_dis_max
            - Max stored energy: e_bat[t]     <= E_cap
            - Battery dynamics:
                  t == 1: e_bat[1] = p_ch_bat[1] * eta_ch - p_dis_bat[1] / eta_dis
                  t  > 1: e_bat[t] = e_bat[t-1] + p_ch_bat[t] * eta_ch - p_dis_bat[t] / eta_dis
            - Power balance:     sum(demand[i,t]) - sum(p_gen[i,t]) - sum(p_wind[i,t])
                                 - p_ch_bat[t] + p_dis_bat[t] == 0

        Returns:
            gp.Model: The Gurobi model after optimization, regardless of solve status.
                Use model.status == GRB.OPTIMAL to check for a valid solution.
        """
        model = gp.Model("model2")
        p_gen = {}
        p_wind = {}
        demand = {}
        p_ch_bat = {}
        p_dis_bat = {}
        e_bat = {}

        # Define Variables
        for t in range(1, self.hours + 1):
            for i in self.P_gens:
                p_gen[i, t] = model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0, name=f"p_gen_{i}_{t}"
                )
            for i in self.P_winds:
                p_wind[i, t] = model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0, name=f"p_wind_{i}_{t}"
                )
            for i in self.Demands["load_distribution"]:
                demand[i, t] = model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0, name=f"demand_{i}_{t}"
                )
            p_ch_bat[t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"p_ch_bat_{t}")
            p_dis_bat[t] = model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name=f"p_dis_bat_{t}"
            )
            e_bat[t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"e_bat_{t}")

        # Define Objective
        model.setObjective(
            gp.quicksum(
                demand[i, t] * self.Demands["bidding_prices"][i]
                for i in self.Demands["load_distribution"]
                for t in range(1, self.hours + 1)
            )
            - gp.quicksum(
                self.P_gens[i]["C"] * p_gen[i, t]
                for i in self.P_gens
                for t in range(1, self.hours + 1)
            ),
            GRB.MAXIMIZE,
        )

        # Define Constraints
        for t in range(1, self.hours + 1):
            # physical constraint conventional generators
            for i in self.P_gens:
                model.addConstr(
                    p_gen[i, t] <= self.P_gens[i]["P_max"],
                    name=f"max_power_Gen_{i}_{t}",
                )
            # physical constraints wind farms
            for i in self.P_winds:
                model.addConstr(
                    p_wind[i, t] <= self.P_winds[i]["P_max"] * self.CF_wind[t],
                    name=f"max_power_wind_{i}_{t}",
                )
            # maximum demand constraint
            for i in self.Demands["load_distribution"]:
                model.addConstr(
                    demand[i, t]
                    <= self.Demands["load_distribution"][i] * self.Load_sum[t],
                    name=f"max_demand_node_{i}_{t}",
                )
            # physical constraints battery
            model.addConstr(
                p_ch_bat[t] <= self.P_ch_max_bat, name=f"max_bat_charge_{t}"
            )
            model.addConstr(
                p_dis_bat[t] <= self.P_dis_max_bat, name=f"max_bat_discharge_{t}"
            )
            model.addConstr(e_bat[t] <= self.E_cap_bat, name=f"max_stored_bat_{t}")
            # temporal dynamics battery
            if t == 1:
                model.addConstr(
                    e_bat[t]
                    == p_ch_bat[t] * self.Effic_ch - p_dis_bat[t] / self.Effic_dis,
                    name=f"e_bat_constr_{t}",
                )
            else:
                model.addConstr(
                    e_bat[t]
                    == e_bat[t - 1]
                    + p_ch_bat[t] * self.Effic_ch
                    - p_dis_bat[t] / self.Effic_dis,
                    name=f"e_bat_constr_{t}",
                )
            # power balance constraint
            model.addConstr(
                gp.quicksum(demand[i, t] for i in self.Demands["load_distribution"])
                - gp.quicksum(p_gen[i, t] for i in self.P_gens)
                - gp.quicksum(p_wind[i, t] for i in self.P_winds)
                + p_ch_bat[t]
                - p_dis_bat[t]
                == 0,
                name=f"power_balance_{t}",
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
        Extracts and computes market results from the optimal multi-period solution.

        Retrieves decision variable values and dual prices from the solved Gurobi
        model and computes key economic metrics across all hours in the horizon.

        Returns:
            dict: A dictionary containing the following market results:

                - market_clearing_price (list of float, length=hours):
                    Dual variable of the power balance constraint at each hour t,
                    representing the system marginal price (SMP) at each t.

                - social_welfare (float):
                    Optimal objective value; total welfare across all hours as
                    the difference between consumer value and generation cost.

                - total_operating_cost (float):
                    Total dispatch cost summed across all generators and hours:
                    sum_t sum_i ( p_gen[i,t] * C[i] ).

                - profit_producers (list of float):
                    Total profit across all hours per production unit, ordered as
                    [gen_0, ..., gen_n, wind_0, ..., wind_m]:
                        Generators: sum_t ( p_gen[i,t] * (MCP[t] - C[i]) )
                        Wind farms:  sum_t ( p_wind[i,t] * MCP[t] )

                - profit_battery (float):
                    Net battery profit across all hours:
                    sum_t ( (p_dis_bat[t] - p_ch_bat[t]) * MCP[t] ),
                    reflecting revenue from discharging minus cost of charging.

                - battery_soc (list of float, length=hours):
                    Battery state of charge at the end of each hour t.

                - battery_charge_volume (list of float, length=hours):
                    Battery charging volume at each hour t (always non-negative).

                - battery_discharge_volume (list of float, length=hours):
                    Battery discharging volume at each hour t with negative sign,
                    so charge/discharge can be plotted on one signed axis.

                - battery_signed_volume (list of float, length=hours):
                    Net signed battery volume at each hour t:
                    p_ch_bat[t] - p_dis_bat[t].
        """
        out_dict = {}
        out_dict["solve_start_perf_counter"] = getattr(self.model, "_solve_start_perf", None)
        out_dict["solve_end_perf_counter"] = getattr(self.model, "_solve_end_perf", None)
        out_dict["solve_time_seconds"] = getattr(self.model, "_solve_time_seconds", None)

        # market clearing price
        out_dict["market_clearing_price"] = [
            self.model.getConstrByName(f"power_balance_{t}").Pi
            for t in range(1, self.hours + 1)
        ]

        # social welfare
        out_dict["social_welfare"] = self.model.ObjVal

        # total operating cost
        out_dict["total_operating_cost"] = sum(
            self.model.getVarByName(f"p_gen_{i}_{t}").X * self.P_gens[i]["C"]
            for i in self.P_gens
            for t in range(1, self.hours + 1)
        )

        # profit of all producers
        out_dict["profit_producers"] = [
            sum(
                self.model.getVarByName(f"p_gen_{i}_{t}").X
                * (out_dict["market_clearing_price"][t - 1] - self.P_gens[i]["C"])
                for t in range(1, self.hours + 1)
            )
            for i in self.P_gens
        ] + [
            sum(
                self.model.getVarByName(f"p_wind_{i}_{t}").X
                * out_dict["market_clearing_price"][t - 1]
                for t in range(1, self.hours + 1)
            )
            for i in self.P_winds
        ]

        # profit of battery & other battery outputs
        out_dict["profit_battery"] = sum(
            (
                self.model.getVarByName(f"p_dis_bat_{t}").X
                - self.model.getVarByName(f"p_ch_bat_{t}").X
            )
            * out_dict["market_clearing_price"][t - 1]
            for t in range(1, self.hours + 1)
        )
        out_dict["battery_soc"] = [
            self.model.getVarByName(f"e_bat_{t}").X for t in range(1, self.hours + 1)
        ]
        out_dict["battery_charge_volume"] = [
            self.model.getVarByName(f"p_ch_bat_{t}").X for t in range(1, self.hours + 1)
        ]
        out_dict["battery_discharge_volume"] = [
            -self.model.getVarByName(f"p_dis_bat_{t}").X
            for t in range(1, self.hours + 1)
        ]
        out_dict["battery_signed_volume"] = [
            self.model.getVarByName(f"p_ch_bat_{t}").X
            - self.model.getVarByName(f"p_dis_bat_{t}").X
            for t in range(1, self.hours + 1)
        ]
        return out_dict

    @staticmethod
    def plot_market_clearing_prices_comparison(
        model_with: "Model2",
        model_without: "Model2",
        model_small: "Model2",
        model_large: "Model2",
        filename: str = "step2_market_price_comparison",
    ) -> None:
        """
        Plot market clearing prices across all hours for four battery scenarios.

        Overlays four time series on a single axes to illustrate
        how battery size affects market clearing price outcomes. Shaded fill regions beneath
        the *without-battery* and *small-battery* curves accentuate their
        deviation from the default case.

        Args:
            model_with (Model2): Solved model with the default battery.
            model_without (Model2): Solved model with no battery.
            model_small (Model2): Solved model with a small battery.
            model_large (Model2): Solved model with a large battery.
            filename (str): Output filename (without extension) written to
                ``results/``. Defaults to ``"step2_price_comparison"``.

        Each model's ``out_dict`` must contain:
            - ``"market_clearing_price"`` (list[float]): Hourly MCP in $/MWh.

        Saves:
            ``results/<filename>.png`` at 150 DPI.
        """
        _, ax = plt.subplots(figsize=(12, 6))

        prices_with = model_with.out_dict["market_clearing_price"]
        prices_without = model_without.out_dict["market_clearing_price"]
        prices_small = model_small.out_dict["market_clearing_price"]
        prices_large = model_large.out_dict["market_clearing_price"]
        x = range(len(prices_with))

        ax.fill_between(x, prices_without, alpha=0.3, color="#2159F3", zorder=1)
        ax.fill_between(x, prices_large, alpha=0.3, color="#FF0000", zorder=1)
        #ax.fill_between(x, prices_small, alpha=0.3, color="#E5BC35", zorder=1)

        # Default battery
        ax.plot(
            x,
            prices_with,
            color="#1C7243",
            linewidth=4,
            marker="o",
            markersize=6,
            label="Default battery",
            zorder=4,
        )

        # Without battery: filled area + line (bottom layer)
        ax.plot(
            x,
            prices_without,
            color="#2159F3",
            linewidth=2,
            marker="o",
            markersize=5,
            label="Without battery",
            zorder=2,
        )

        # Small battery
        ax.plot(
            x,
            prices_small,
            color="#E5BC35",
            linewidth=1,
            marker="x",
            markersize=3,
            label="Small battery",
            zorder=5,
        )

        # Large battery
        ax.plot(
            x,
            prices_large,
            color="#FF0000",
            linewidth=2,
            marker="o",
            markersize=5,
            label="Large battery",
            zorder=3,
        )

        ax.set_xlabel("Hour (t)", fontsize=21, labelpad=10)
        ax.set_ylabel("Market Clearing Price ($/MWh)", fontsize=21, labelpad=10)
        ax.set_xticks(x)
        ax.set_yticks(range(12, 21, 2))
        ax.set_yticklabels([str(v) for v in range(12, 21, 2)])
        ax.set_ylim(10, 21)
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(fontsize=16, framealpha=0.9, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="grey")
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(f"results/{filename}.png", dpi=150)

    @staticmethod
    def plot_profit_producers_comparison(
        model_with: "Model2",
        model_without: "Model2",
        model_small: "Model2",
        model_large: "Model2",
        filename: str = "step2_profit_producers_comparison",
    ) -> None:
        """
        Plot producer profits for four battery scenarios side-by-side.

        Renders a grouped bar chart with one cluster per producer, comparing
        profits across the default, no-battery, small, and large battery
        scenarios. Conventional generators are labelled G1, G2, … and wind
        farms W1, W2, …. Dollar values are annotated above each non-zero bar.

        Args:
            model_with (Model2): Model2 - default battery.
            model_without (Model2): Model2 — no battery.
            model_small (Model2): Model2 — small battery.
            model_large (Model2): Model2 — large battery.
            filename (str): Output filename without extension, written to
                ``results/``. Defaults to ``"step2_profit_producers"``.

        Saves:
            ``results/<filename>.png`` at 150 DPI.
        """
        profits_with = model_with.out_dict["profit_producers"]
        profits_without = model_without.out_dict["profit_producers"]
        profits_small = model_small.out_dict["profit_producers"]
        profits_large = model_large.out_dict["profit_producers"]
        n = len(profits_with)
        x = np.arange(n)
        width = 0.2

        scenario_colors = {
            "Default battery": "#1C7243",
            "Without battery": "#2196F3",
            "Small battery": "#FF0000",
            "Large battery": "#E5BC35",
        }
        all_profits = [profits_with, profits_without, profits_small, profits_large]
        offsets = [-1.5, -0.5, 0.5, 1.5]

        labels = [f"W{i - n + 7}" if i >= n - 6 else f"G{i + 1}" for i in range(n)]
        gen_colors = ["#2196F3" if i >= n - 6 else "#4CAF50" for i in range(n)]

        _, ax = plt.subplots(figsize=(16, 6))

        for (scenario, color), profits, offset in zip(
            scenario_colors.items(), all_profits, offsets
        ):
            bars = ax.bar(
                x + offset * width,
                profits,
                width=width,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                label=scenario,
                zorder=2,
            )
            
        ax.set_xlabel("Producer", fontsize=23, labelpad=10)
        ax.set_ylabel("Profit ($)", fontsize=23, labelpad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.tick_params(axis="both", labelsize=17)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="grey", zorder=1)
        ax.set_axisbelow(True)

        # Legend: scenarios (line style) + producer type (colour patch)
        scenario_handles, scenario_labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=scenario_handles,
            fontsize=18,
            framealpha=0.9,
            loc="upper left",
        )

        plt.tight_layout()
        plt.savefig(f"results/{filename}.png", dpi=150)
