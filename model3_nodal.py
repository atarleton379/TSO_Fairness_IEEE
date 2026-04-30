import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import time

from src.input import InputHandler


class Model3Nodal:
    """
    Single-hour day-ahead market clearing incl. network constraints
    """

    def __init__(self, hour: int, inp_hndl: InputHandler, sensitivity_factor: float):
        """
        Initializes and runs the network-constrained optimal power flow model
        for a given hour. Nodal mdodel.

        Args:
            hour (int): The hour of the day for which the model is run.
            inp_hndl (InputHandler): An InputHandler instance containing all
                necessary input data, including generators, wind farms,
                capacity factors, demand, and network topology.
            sensitivity_factor (float): Power Transfer Distribution Factor (PTDF)
                or equivalent sensitivity factor used to model power flow
                across transmission lines.

        Attributes:
            sensitivity_factor (float): Stored PTDF sensitivity factor.
            hour (int): Stored hour index.
            P_gens (dict): Conventional generator data from the input handler.
            P_winds (dict): Wind farm data from the input handler.
            CF_wind (dict): Wind capacity factors from the input handler.
            Demands (dict): Full demand data dictionary from the input handler.
            Load_sum (dict): System load extracted from the demands dictionary.
            Lines (dict): Transmission line data from the input handler.
            Nodes (list): Network node data from the input handler.
            model (gurobipy.Model): The Gurobi model after optimization.
            out_dict (dict): Output analysis results, set only if the model
                reaches an optimal solution.
        """
        self.sensitivity_factor = sensitivity_factor
        self.hour = hour
        self.P_gens = inp_hndl.generators
        self.P_winds = inp_hndl.wind_farms
        self.CF_wind = inp_hndl.CF_wind
        self.Demands = inp_hndl.demands
        self.Load_sum = inp_hndl.demands["system_load"]
        self.Lines = inp_hndl.lines
        self.Nodes = inp_hndl.nodes
        self.Sys_base = inp_hndl.sys_base

        self.model = self._run_model()
        if self.model.status == gp.GRB.OPTIMAL:
            self.out_dict = self._analyze_output()

    def _run_model(self) -> gp.Model:
        """
        Builds and solves the network-constrained linearized DC optimal power flow (DC-OPF)
        model for the current hour.

        Extends the copper-plate market clearing formulation by explicitly
        modeling the transmission network using a DC power flow approximation,
        enabling nodal energy pricing (LMPs) and congestion management.

        Decision Variables:
            p_gen[i]    (continuous, >= 0):    Power output of conventional generator i.
            p_wind[i]   (continuous, >= 0):    Power output of wind farm i.
            demand[i]   (continuous, >= 0):    Served demand for load i.
            p_line[i]   (continuous, unconstrained): Power flow on transmission line i.
            v_angle[i]  (continuous, unconstrained): Voltage angle at node i (radians).

        Objective:
            Maximize social welfare:
                sum_i(demand[i] * bidding_price[i]) - sum_i(C[i] * p_gen[i])

        Constraints:
            - Max generation:    p_gen[i]  <= P_max[i]                           for all i in P_gens
            - Max wind output:   p_wind[i] <= P_max[i] * CF_wind[hour]           for all i in P_winds
            - Max demand:        demand[i] <= load_distribution[i] * Load[hour]  for all i in Demands
            - Line capacity:     -F_max[i] * sf <= p_line[i] <= F_max[i] * sf   for all i in Lines
            - DC power flow:     p_line[i] == (1/x[i]) * (theta_from - theta_to) for all i in Lines
            - Nodal balance:     sum(inflows) - sum(outflows)
                                 + sum(generation) + sum(wind) - sum(demand) == 0  for all n in Nodes

        Returns:
            gp.Model: The Gurobi model after optimization, regardless of solve status.
                Use model.status == GRB.OPTIMAL to check for a valid solution.
        """
        model = gp.Model("model1")

        p_gen = {}
        p_wind = {}
        demand = {}
        p_line = {}
        v_angle = {}

        # Define Variables
        for i in self.P_gens:
            p_gen[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"p_gen_{i}")
        for i in self.P_winds:
            p_wind[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"p_wind_{i}")
        for i in self.Demands["load_distribution"]:
            demand[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"demand_{i}")
        for i in self.Lines:
            p_line[i] = model.addVar(
                vtype=GRB.CONTINUOUS, lb=-float("inf"), name=f"p_line_{i}"
            )
        for i in self.Nodes:
            v_angle[i] = model.addVar(
                vtype=GRB.CONTINUOUS, lb=-float("inf"), name=f"v_angle_{i}"
            )
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
        # line constraints
        for i in p_line:
            # transmission capacity constraints
            model.addConstr(
                p_line[i] <= (self.Lines[i]["capacity"] * self.sensitivity_factor),
                name=f"transmission_capacity_line_{i}_ub",
            )
            model.addConstr(
                (-self.Lines[i]["capacity"] * self.sensitivity_factor) <= p_line[i],
                name=f"transmission_capacity_line_{i}_lb",
            )
            # simplified power flow (DC: p = (System base/x) * (theta_from - theta_to))
            model.addConstr(
                p_line[i]
                == (self.Sys_base / self.Lines[i]["x"])
                * (v_angle[self.Lines[i]["from"]] - v_angle[self.Lines[i]["to"]]),
                name=f"dc_power_flow_{i}",
            )
        # nodal balance constraints
        for n in self.Nodes:
            model.addConstr(
                gp.quicksum(
                    p_line[l] for l in self.Lines if self.Lines[l]["to"] == n
                )  # lines flowing into node n
                - gp.quicksum(
                    p_line[l] for l in self.Lines if self.Lines[l]["from"] == n
                )  # lines flowing out of node n
                + gp.quicksum(
                    p_gen[g] for g in self.P_gens if self.P_gens[g]["node"] == n
                )  # generation at node n
                + gp.quicksum(
                    p_wind[w] for w in self.P_winds if self.P_winds[w]["node"] == n
                )  # wind generation at node n
                - gp.quicksum(
                    demand[d]
                    for d in self.Demands["load_location"]
                    if self.Demands["load_location"][d] == n
                )  # load at node n
                == 0,
                name=f"power_balance_{n}",
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
        Extracts and computes nodal market results from the optimal DC-OPF solution.

        Retrieves primal variable values and dual prices (LMPs) from the solved
        Gurobi model and computes a comprehensive set of economic metrics,
        including nodal prices, congestion rent, producer profits, and consumer
        utility at each node.

        Returns:
            dict: A dictionary containing the following market results:

                - node_{n}_price (float):
                    Locational marginal price (LMP) at each node n.

                - social_welfare (float):
                    Optimal objective value; total welfare across the network.

                - line_{i}_flow (float):
                    Realized power flow on transmission line i (MW).

                - total_congestion_rent (float):
                    Revenue collected by the network operator:
                    sum(LMP[n] * demand[n]) - sum(LMP[n] * p_gen[n]) - sum(LMP[n] * p_wind[n]).

                - total_operating_cost (float):
                    Total dispatch cost across all conventional generators:
                    sum_i( p_gen[i] * C[i] ).

                - profit_producers (list of float):
                    Per-unit profit ordered as [gen_0, ..., gen_n, wind_0, ..., wind_m]:
                        Generators: p_gen[i] * (LMP[node_i] - C[i])
                        Wind farms:  p_wind[i] * LMP[node_i]

                - utility_demands (list of float):
                    Per-load consumer utility:
                    demand[i] * (bidding_price[i] - LMP[node_i])  for each load i.

                - demands (list of float):
                    Cleared demand quantity for each load i.

                - total_power_consumed (float):
                    Sum of all cleared demands across the network.

                - gen_profits (dict):
                    Individual profit for each conventional generator i.

                - total_gen_profit (float):
                    Sum of profits across all conventional generators.

                - avg_gen_profit (float):
                    Average profit per conventional generator.

                - wind_profits (dict):
                    Individual profit for each wind farm i.

                - total_wind_profit (float):
                    Sum of profits across all wind farms.

                - avg_wind_profit (float):
                    Average profit per wind farm.
        """
        out_dict = {}
        out_dict["solve_start_perf_counter"] = getattr(self.model, "_solve_start_perf", None)
        out_dict["solve_end_perf_counter"] = getattr(self.model, "_solve_end_perf", None)
        out_dict["solve_time_seconds"] = getattr(self.model, "_solve_time_seconds", None)

        # nodal prices
        for n in self.Nodes:
            out_dict[f"node_{n}_price"] = self.model.getConstrByName(
                f"power_balance_{n}"
            ).Pi

        # social welfare
        out_dict["social_welfare"] = self.model.ObjVal

        # total congestion rent
        out_dict["total_congestion_rent"] = (
            sum(
                (
                    -self.model.getConstrByName(
                        f"power_balance_{self.Demands['load_location'][i]}"
                    ).Pi
                )
                * self.model.getVarByName(f"demand_{i}").X
                for i in self.Demands["load_location"]
            )
            - sum(
                (
                    -self.model.getConstrByName(
                        f"power_balance_{self.P_gens[i]['node']}"
                    ).Pi
                )
                * self.model.getVarByName(f"p_gen_{i}").X
                for i in self.P_gens
            )
            - sum(
                (
                    -self.model.getConstrByName(
                        f"power_balance_{self.P_winds[i]['node']}"
                    ).Pi
                )
                * self.model.getVarByName(f"p_wind_{i}").X
                for i in self.P_winds
            )
        )

        # total operating cost
        out_dict["total_operating_cost"] = sum(
            self.model.getVarByName(f"p_gen_{i}").X * self.P_gens[i]["C"]
            for i in self.P_gens
        )

        # profit of all producers
        out_dict["profit_producers"] = [
            self.model.getVarByName(f"p_gen_{i}").X
            * (
                -self.model.getConstrByName(
                    f"power_balance_{self.P_gens[i]['node']}"
                ).Pi
                - self.P_gens[i]["C"]
            )
            for i in self.P_gens
        ] + [
            self.model.getVarByName(f"p_wind_{i}").X
            * (
                -self.model.getConstrByName(
                    f"power_balance_{self.P_winds[i]['node']}"
                ).Pi
            )
            for i in self.P_winds
        ]

        # utility of demand
        out_dict["utility_demands"] = [
            self.model.getVarByName(f"demand_{i}").X
            * (
                self.Demands["bidding_prices"][i]
                - (
                    -self.model.getConstrByName(
                        f"power_balance_{self.Demands['load_location'][i]}"
                    ).Pi
                )
            )
            for i in self.Demands["load_location"]
        ]

        # overview of demand
        out_dict["demands"] = [
            self.model.getVarByName(f"demand_{i}").X
            for i in self.Demands["load_location"]
        ]
        # total demand
        out_dict["total_power_consumed"] = sum(out_dict["demands"])

        # Conventional generator profits (individual, total, average)
        gen_profits = {
            i: self.model.getVarByName(f"p_gen_{i}").X
            * (
                -self.model.getConstrByName(
                    f"power_balance_{self.P_gens[i]['node']}"
                ).Pi
                - self.P_gens[i]["C"]
            )
            for i in self.P_gens
        }
        out_dict["gen_profits"] = gen_profits
        out_dict["total_gen_profit"] = sum(gen_profits.values())
        out_dict["avg_gen_profit"] = out_dict["total_gen_profit"] / len(gen_profits)

        # Wind generator profits (individual, total, average)
        wind_profits = {
            i: self.model.getVarByName(f"p_wind_{i}").X
            * (
                -self.model.getConstrByName(
                    f"power_balance_{self.P_winds[i]['node']}"
                ).Pi
            )
            for i in self.P_winds
        }
        out_dict["wind_profits"] = wind_profits
        out_dict["total_wind_profit"] = sum(wind_profits.values())
        out_dict["avg_wind_profit"] = out_dict["total_wind_profit"] / len(wind_profits)

        # line flows
        for i in self.Lines:
            out_dict[f"line_{i}_flow"] = self.model.getVarByName(f"p_line_{i}").X
        return out_dict

    @staticmethod
    def plot_nodal_price_variance(sens_data):
        """
        Plot the nodal price spread (max - min) across sensitivity scenarios.

        Iterates over sensitivity runs, skipping any zonal scenarios, and computes
        the spread between the highest and lowest nodal prices for each line
        capacity factor. Results are displayed as a bar chart and saved to disk.

        Args:
            sens_data (dict): Nested dictionary where outer keys are scenario names
                (e.g. 'model3_nodal_sens_0.8') and inner dicts map node keys
                ('node_*_price') to their price values.

        Saves:
            results/step3_nodal_price_variance.png
        """
        spread_results = {}
        for factor, out_dict in sens_data.items():
            if "zonal" in factor:
                continue
            nodal_prices = [
                -v
                for k, v in out_dict.items()
                if k.startswith("node_") and k.endswith("_price")
            ]
            spread_results[factor] = max(nodal_prices) - min(nodal_prices)
            mean = sum(nodal_prices) / len(nodal_prices)
        sorted_factors = sorted(spread_results.keys())
        x_labels = [f"{float(f.split('_')[-1])*100:.0f}%" for f in sorted_factors]
        spread_values = [spread_results[f] for f in sorted_factors]

        _, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(x_labels, spread_values, color="darkorange", edgecolor="black")
        ax.set_xlabel("Line Capacity Available (%)", fontsize=15, labelpad=10)
        ax.set_ylabel("Nodal Price Variance (€/MWh)²", fontsize=15, labelpad=10)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_ylim(0, max(spread_values) * 1.15)
        plt.tight_layout()
        plt.savefig("results/step3_nodal_price_variance.png", dpi=150)
