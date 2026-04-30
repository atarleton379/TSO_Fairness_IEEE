import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

from src.input import InputHandler


class Model5:
    """
    Single-hour day-ahead market clearing followed by single-hour balancing market clearing.
    """

    def __init__(self, hour: int, inp_hndl: InputHandler):
        """
        Initializes and runs a two-settlement market model for a given hour,
        consisting of a Day-Ahead Market (DAM) clearing followed by a
        real-time Balancing Market (BM) clearing.

        The DAM clears scheduled dispatch based on forecasted conditions.
        The BM then corrects any imbalances between scheduled and actual
        output (i.e., generator failure and wind forecast errors), procuring upward or downward
        balancing from available providers.

        Args:
            hour (int): The hour of the day for which the model is run.
            inp_hndl (InputHandler): An InputHandler instance containing all
                necessary input data, including generators, wind farms,
                capacity factors, and demand information.

        Attributes:
            hour (int): Stored hour index.
            P_gens (dict): Conventional generator data from the input handler.
            P_winds (dict): Wind farm data from the input handler.
            CF_wind (dict): Wind capacity factors from the input handler.
            Demands (dict): Full demand data dictionary from the input handler.
            Load_sum (dict): System load extracted from the demands dictionary.
            DAC (gurobipy.Model): Solved Day-Ahead Market clearing model.
            BMC (gurobipy.Model): Solved Balancing Market clearing model.
            BSP (dict): Balancing service providers or related balancing data.
            p_gen_actual (dict): Realized conventional generator dispatch
                after balancing.
            p_wind_actual (dict): Realized wind farm output after balancing.
            out_dict (dict): Combined DAM and BM output analysis results,
                set only if both models reach an optimal solution.
        """
        self.hour = hour
        self.P_gens = inp_hndl.generators
        self.P_winds = inp_hndl.wind_farms
        self.CF_wind = inp_hndl.CF_wind
        self.Demands = inp_hndl.demands
        self.Load_sum = inp_hndl.demands["system_load"]

        self.day_ahead_model = self._run_model_day_ahead_clearing()

        (
            self.balancing_model,
            self.BSP,
            self.p_gen_actual,
            self.p_wind_actual,
            self.C_plus,
            self.C_minus,
            self.C_curt,
        ) = self._run_model_balance_market_clearing()

        if (
            self.day_ahead_model.status == gp.GRB.OPTIMAL
            and self.balancing_model.status == gp.GRB.OPTIMAL
        ):
            self.out_dict = self._analyze_output()
            self._plot_merit_order()
            self._plot_revenue_breakdown()

    def _run_model_day_ahead_clearing(self) -> gp.Model:
        """
        Builds and solves the Day-Ahead Market (DAM) clearing model for the
        current hour.

        Clears the market based on forecasted wind capacity factors and demand,
        producing scheduled dispatch quantities and a day-ahead market clearing
        price (DA-MCP). These schedules serve as the reference point for
        subsequent balancing market corrections.

        Decision Variables:
            p_gen[i]  (continuous, >= 0): Scheduled output of conventional generator i.
            p_wind[i] (continuous, >= 0): Scheduled output of wind farm i.
            demand[i] (continuous, >= 0): Scheduled served demand for load i.

        Objective:
            Maximize social welfare:
                sum_i(demand[i] * bidding_price[i]) - sum_i(C[i] * p_gen[i])

        Constraints:
            - Max generation:  p_gen[i]  <= P_max[i]                           for all i in P_gens
            - Max wind output: p_wind[i] <= P_max[i] * CF_wind[hour]           for all i in P_winds
            - Max demand:      demand[i] <= load_distribution[i] * Load[hour]  for all i in Demands
            - Power balance:   sum(demand) - sum(p_gen) - sum(p_wind) == 0

        Returns:
            gp.Model: The solved DAM Gurobi model, regardless of solve status.
                Use model.status == GRB.OPTIMAL to check for a valid solution.
                Scheduled quantities are accessible via model.getVarByName().
                DA-MCP is accessible as the dual of the "power_balance" constraint.
        """

        model = gp.Model("DAM_5")
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

    def _run_model_balance_market_clearing(self) -> tuple[gp.Model, list, dict, dict]:
        """
        Builds and solves the real-time Balancing Market (BM) clearing model,
        correcting deviations between DAM schedules and actual realized output.

        Retrieves DAM schedules from the solved DAC model, applies predefined
        real-time deviations (generator failure, wind forecast errors), then
        procures the minimum-cost balancing response from available Balancing
        Service Providers (BSPs).

        Real-time Deviations (hardcoded):
            - Generator 9:        Unexpected failure — output set to 0.
            - Wind farms 1 & 2:   Produce 18% less than DAM schedule.
            - Wind farms 3–6:     Produce 15% more than DAM schedule.

        Balancing Service Providers (BSPs):
            Generators 1–7 are eligible to provide upward and downward
            regulation. Their balancing bid prices are derived from the
            DA-MCP and their individual production costs:
                C_up[j]   = DA-MCP + 0.10 * C[j]   (upward regulation bid)
                C_down[j] = DA-MCP - 0.15 * C[j]   (downward regulation bid)

        Decision Variables:
            b_gen_up[j]     (continuous, >= 0): Upward regulation from BSP j.
            b_gen_down[j]   (continuous, >= 0): Downward regulation from BSP j.
            b_demand_curt[i](continuous, >= 0): Demand curtailment for load i
                                                at C_curt = $500/MWh.

        Objective:
            Minimize total balancing cost:
                sum_j(C_up[j] * b_gen_up[j])
                - sum_j(C_down[j] * b_gen_down[j])
                + sum_i(C_curt * b_demand_curt[i])

        Constraints:
            - Down activation:  b_gen_down[j] <= p_gen_DAM[j]                    for all j in BSP
            - Up activation:    b_gen_up[j]   <= P_max[j] - p_gen_actual[j]      for all j in BSP
            - BM balance:       sum(b_gen_up) - sum(b_gen_down)
                                + sum(p_gen_actual) + sum(p_wind_actual)
                                + sum(b_demand_curt) == sum(demand_actual)

        Returns:
            tuple:
                - model (gp.Model):   Solved balancing market Gurobi model.
                - BSP (list):             List of BSP generator keys (generators 1–7).
                - p_gen_actual (dict):    Realized conventional generator output
                                          after applying generator failure.
                - p_wind_actual (dict):   Realized wind farm output after applying
                                          forecast errors.
        """
        # extract cleared results from day ahead market
        p_gen_DAM = {
            i: self.day_ahead_model.getVarByName(f"p_gen_{i}").X for i in self.P_gens
        }
        p_wind_DAM = {
            i: self.day_ahead_model.getVarByName(f"p_wind_{i}").X for i in self.P_winds
        }
        demand_DAM = {
            i: self.day_ahead_model.getVarByName(f"demand_{i}").X
            for i in self.Demands["load_distribution"]
        }

        # Change input data according to assignment guidelines
        BSP = list(self.P_gens.keys())[
            :7
        ]  # generator 1-7 are potential BSPs (Balancing service providers), executive decision of project group
        P_gen_max = {j: self.P_gens[j]["P_max"] for j in BSP}
        C_plus = {
            j: self.day_ahead_model.getConstrByName("power_balance").Pi
            + self.P_gens[j]["C"] * 0.1
            for j in BSP
        }  # DAM clearing price + 10% production costs
        C_minus = {
            j: self.day_ahead_model.getConstrByName("power_balance").Pi
            - self.P_gens[j]["C"] * 0.15
            for j in BSP
        }  # DAM clearing price - 15% production costs
        C_curt = 500  # $/MWh

        # implement difference between scheduled and actual production of wind farms and generators
        demand_actual = demand_DAM.copy()
        p_gen_actual = p_gen_DAM.copy()
        if 9 in p_gen_actual:  # generator 9 has an unexpected failure
            p_gen_actual[9] = 0

        p_wind_actual = {}
        for i in p_wind_DAM:
            if i == 1 or i == 2:
                p_wind_actual[i] = (
                    p_wind_DAM[i] - p_wind_DAM[i] * 0.18
                )  # windfarms 1 and 2 produce 18% less than forecasted
            else:
                p_wind_actual[i] = (
                    p_wind_DAM[i] + p_wind_DAM[i] * 0.15
                )  # windfarms 3-6 produce 15% more than forecasted

        # Construct optimization model
        model = gp.Model("Balancing_market")
        b_gen_up = {}
        b_gen_down = {}
        b_demand_curt = {}

        # Define Variables
        for j in BSP:
            b_gen_up[j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"b_gen_up_{j}")
            b_gen_down[j] = model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name=f"b_gen_down_{j}"
            )
        for i in self.Demands:
            b_demand_curt[i] = model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name=f"b_demand_{i}"
            )

        # Define Objective
        model.setObjective(
            gp.quicksum(C_plus[j] * b_gen_up[j] for j in BSP)
            - gp.quicksum(C_minus[j] * b_gen_down[j] for j in BSP)
            + gp.quicksum(C_curt * b_demand_curt[i] for i in self.Demands),
            GRB.MINIMIZE,
        )

        # Define Constraints
        # Physical down regulation constraint of BSPs
        model.addConstrs(
            (b_gen_down[j] <= p_gen_DAM[j] for j in BSP), name="down_activation"
        )  # downward regulation can't be more than what was scheduled in DAM (assuming P_min=0)

        # Physical up regulation constraint of BSPs
        model.addConstrs(
            (b_gen_up[j] <= P_gen_max[j] - p_gen_actual[j] for j in BSP),
            name="up_activation",
        )  # upward regulation can't be more than difference between P_max and DAM schedule

        # Power balance constraint of balancing market
        model.addConstr(
            gp.quicksum(b_gen_up[j] for j in BSP)
            - gp.quicksum(b_gen_down[j] for j in BSP)
            + sum(p_gen_actual.values())
            + sum(p_wind_actual.values())
            + sum(b_demand_curt.values())
            == sum(demand_actual.values()),
            name="BMC_balance",
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
        return model, BSP, p_gen_actual, p_wind_actual, C_plus, C_minus, C_curt

    def _analyze_output(self) -> dict:
        """
        Extracts and computes combined Day-Ahead and Balancing Market results
        from both optimal settlement models.

        Retrieves primal and dual values from both the DAM (self.day_ahead_model) and
        Balancing Market (self.balancing_model) models, computing economic outcomes for
        all market participants under both one-price and two-price imbalance
        settlement schemes.

        Imbalance Settlement Schemes:
            One-price: All imbalances settled at BM clearing price (BCM).
            Two-price: Imbalances that worsen the system balance are settled
                       at BCM; imbalances that help the system balance are
                       settled at DA-MCP. Specifically, BCM is applied when:
                           - BCM > DA-MCP and imbalance < 0 (short position in high price)
                           - BCM < DA-MCP and imbalance > 0 (long position in low price)
                       DA-MCP applies otherwise. This prevents intentional imbalancing of system, specifically by wind farms.

        Returns:
            dict: A nested dictionary with three top-level sections:

                day_ahead_market:
                    - market_clearing_price_DAM (float):
                        Dual of the DAM power balance constraint (DA-MCP).
                    - social_welfare (float):
                        DAM optimal objective value.
                    - total_operating_cost (float):
                        sum_i( p_gen_DAM[i] * C[i] ).
                    - scheduled_power_output (dict):
                        DAM dispatch schedule keyed as "gen_{i}" and "wind_{i}".
                    - profit_producers_DAM (dict):
                        Per-unit DAM profit:
                            Generators: p_gen[i] * (DA-MCP - C[i])
                            Wind farms:  p_wind[i] * DA-MCP
                    - utility_demands (list of float):
                        Per-load consumer utility:
                            demand[i] * (bidding_price[i] - DA-MCP)

                balancing_market:
                    - market_clearing_price_BM (float):
                        Dual of the BM balance constraint (BCM).
                    - upwards_regulation_schedule (dict):
                        Activated upward regulation per BSP j.
                    - downwards_regulation (dict):
                        Activated downward regulation per BSP j.
                    - curtailed_demand (dict):
                        Curtailed demand volume per load i.
                    - revenue_BSPs (dict):
                        Revenue per BSP from energy activation:
                            (b_gen_up[j] + b_gen_down[j]) * BCM
                    - imbalance_settlements_one_price (dict):
                        Per-unit imbalance settlement under one-price scheme:
                            (actual[i] - scheduled[i]) * BCM
                    - imbalance_settlements_two_price (dict):
                        Per-unit imbalance settlement under two-price scheme,
                        applying BCM or DA-MCP depending on imbalance direction
                        relative to system need.

                total_results:
                    - production_schedule_at_delivery (dict):
                        Realized output at time of delivery, keyed as
                        "gen_{i}" and "wind_{i}".
                    - total_revenues_one_price (dict):
                        Combined revenue per unit under one-price settlement:
                            DAM profit + imbalance settlement + BSP revenue
                    - total_revenues_two_price (dict):
                        Combined revenue per unit under two-price settlement:
                            DAM profit + imbalance settlement + BSP revenue
        """
        # Day Ahead Market
        DAM_clearing = self.day_ahead_model.getConstrByName("power_balance").Pi
        social_welfare = self.day_ahead_model.ObjVal

        DAM_schedule = {
            f"gen_{i}": self.day_ahead_model.getVarByName(f"p_gen_{i}").X
            for i in self.P_gens
        } | {
            f"wind_{i}": self.day_ahead_model.getVarByName(f"p_wind_{i}").X
            for i in self.P_winds
        }
        profit_producers_DAM = {
            f"gen_{i}": self.day_ahead_model.getVarByName(f"p_gen_{i}").X
            * (DAM_clearing - self.P_gens[i]["C"])
            for i in self.P_gens
        } | {
            f"wind_{i}": self.day_ahead_model.getVarByName(f"p_wind_{i}").X
            * DAM_clearing
            for i in self.P_winds
        }
        utility_demands = [
            self.day_ahead_model.getVarByName(f"demand_{i}").X
            * (self.Demands["bidding_prices"][i] - DAM_clearing)
            for i in self.Demands["load_distribution"]
        ]
        total_cost_DAM = sum(
            self.day_ahead_model.getVarByName(f"p_gen_{i}").X * self.P_gens[i]["C"]
            for i in self.P_gens
        )

        # Balancing Market
        BCM_clearing = self.balancing_model.getConstrByName("BMC_balance").Pi
        # imbalance settlements #for all involuntary production differences
        # under one-price scheme
        imbalance_settlement_one_price = {
            f"gen_{i}": (
                self.p_gen_actual[i] - self.day_ahead_model.getVarByName(f"p_gen_{i}").X
            )
            * BCM_clearing
            for i in self.P_gens
        } | {
            f"wind_{i}": (
                self.p_wind_actual[i]
                - self.day_ahead_model.getVarByName(f"p_wind_{i}").X
            )
            * BCM_clearing
            for i in self.P_winds
        }

        # under two-price scheme
        imbalance_settlement_two_price = {
            f"gen_{i}": (
                (
                    imb := self.p_gen_actual[i]
                    - self.day_ahead_model.getVarByName(f"p_gen_{i}").X
                )
                * (
                    BCM_clearing
                    if (
                        (BCM_clearing > DAM_clearing and imb < 0)
                        or (BCM_clearing < DAM_clearing and imb > 0)
                    )
                    else DAM_clearing
                )
            )
            for i in self.P_gens
        } | {
            f"wind_{i}": (
                (
                    imb := self.p_wind_actual[i]
                    - self.day_ahead_model.getVarByName(f"p_wind_{i}").X
                )
                * (
                    BCM_clearing
                    if (
                        (BCM_clearing > DAM_clearing and imb < 0)
                        or (BCM_clearing < DAM_clearing and imb > 0)
                    )
                    else DAM_clearing
                )
            )
            for i in self.P_winds
        }

        # BSP behaviour on energy activation
        BCM_schedule_upward = {
            j: self.balancing_model.getVarByName(f"b_gen_up_{j}").X for j in self.BSP
        }
        BCM_schedule_downward = {
            j: self.balancing_model.getVarByName(f"b_gen_down_{j}").X for j in self.BSP
        }
        BCM_schedule_curt_demand = {
            i: self.balancing_model.getVarByName(f"b_demand_{i}").X for i in self.Demands
        }
        BCM_revenue = {
            j: (
                self.balancing_model.getVarByName(f"b_gen_up_{j}").X
                + self.balancing_model.getVarByName(f"b_gen_down_{j}").X
            )
            * BCM_clearing
            for j in self.BSP
        }  # for all BSPs participating in energy activation market

        # combined results
        actual_prod = {  # actual schedule at time of delivery
            f"gen_{i}": self.p_gen_actual[i] for i in self.p_gen_actual
        } | {f"wind_{i}": self.p_wind_actual[i] for i in self.p_wind_actual}
        total_revenues_one_price = {
            i: profit_producers_DAM.get(i, 0)
            + imbalance_settlement_one_price.get(i, 0)
            + BCM_revenue.get(i, 0)
            for i in list(profit_producers_DAM.keys())
            + list(imbalance_settlement_one_price.keys())
        }

        total_revenues_two_price = {
            i: profit_producers_DAM.get(i, 0)
            + imbalance_settlement_two_price.get(i, 0)
            + BCM_revenue.get(i, 0)
            for i in list(profit_producers_DAM.keys())
            + list(imbalance_settlement_two_price.keys())
        }

        # create output json
        out_dict = {}
        out_dict["day_ahead_market"] = {
            "solve_start_perf_counter": getattr(self.day_ahead_model, "_solve_start_perf", None),
            "solve_end_perf_counter": getattr(self.day_ahead_model, "_solve_end_perf", None),
            "solve_time_seconds": getattr(self.day_ahead_model, "_solve_time_seconds", None),
            "market_clearing_price_DAM": DAM_clearing,  # market clearing price
            "social_welfare": social_welfare,  # social welfare
            "total_operating_cost": total_cost_DAM,
            "scheduled_power_output": DAM_schedule,
            "profit_producers_DAM": profit_producers_DAM,
            "utility_demands": utility_demands,
        }

        out_dict["balancing_market"] = {
            "solve_start_perf_counter": getattr(self.balancing_model, "_solve_start_perf", None),
            "solve_end_perf_counter": getattr(self.balancing_model, "_solve_end_perf", None),
            "solve_time_seconds": getattr(self.balancing_model, "_solve_time_seconds", None),
            "market_clearing_price_BM": BCM_clearing,
            "upwards_regulation_schedule": BCM_schedule_upward,
            "downwards_regulation": BCM_schedule_downward,
            "curtailed_demand": BCM_schedule_curt_demand,
            "revenue_BSPs": BCM_revenue,  # for voluntary BSPs
            "imbalance_settlements_one_price": imbalance_settlement_one_price,  # for involuntary changes between forecast and actual production
            "imbalance_settlements_two_price": imbalance_settlement_two_price,
        }

        out_dict["total_results"] = {
            "production_schedule_at_delivery": actual_prod,
            "total_revenues_one_price": total_revenues_one_price,
            "total_revenues_two_price": total_revenues_two_price,
        }
        return out_dict

    def _plot_merit_order(self):
        """
        Plot the upward merit order curve for the balancing market.

        Constructs an upward bid stack from available generator headroom
        (P_max minus actual dispatch) and demand curtailment capacity, sorts
        bids by price, and renders them as a bar chart with a step curve overlay.

        Uses a broken y-axis to handle the large price gap between conventional
        generator bids and the curtailment price (C_curt). The bottom axis shows
        the normal bid range; the top axis isolates the curtailment block.

        Annotations include:
            - A horizontal dashed line at the balancing market clearing price (BM).
            - A vertical dashed line at the mismatch volume (net demand not served
              by day-ahead dispatch).
            - A cross marker at the clearing point where the two lines intersect.
            - Bar colors distinguishing dispatched (green), undispatched (gray),
              and curtailment (red hatched) blocks.

        Output is saved to 'results/step5_merit_order.png' at 150 dpi.
        """
        bm_price = self.balancing_model.getConstrByName("BMC_balance").Pi

        mismatch_volume = (
            sum(
                self.day_ahead_model.getVarByName(f"demand_{i}").X
                for i in self.Demands["load_distribution"]
            )
            - sum(self.p_gen_actual.values())
            - sum(self.p_wind_actual.values())
        )

        # Build upward bid stack (merit order)
        entries = []  # (price, volume, label)
        for j in self.BSP:
            vol = self.P_gens[j]["P_max"] - self.p_gen_actual[j]
            if vol > 1e-6:
                entries.append((self.C_plus[j], vol, j))

        # Add demand curtailment as highest-price block
        total_curt_cap = sum(
            self.day_ahead_model.getVarByName(f"demand_{i}").X
            for i in self.Demands["load_distribution"]
        )
        entries.append((self.C_curt, total_curt_cap, "curt"))

        entries.sort(key=lambda x: x[0])

        # Colors
        color_dispatched = "#4CAF50"
        color_undispatched = "#BDBDBD"
        color_curt = "#E53935"
        color_bm_line = "#E53935"
        color_mismatch = "#FF6F00"

        # Broken y-axis setup
        lower_max = max(e[0] for e in entries if e[2] != "curt") * 1.25
        upper_min = self.C_curt * 0.85
        upper_max = self.C_curt * 1.10

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [1, 3]}
        )
        fig.subplots_adjust(hspace=0.08)

        # Draw bars and step curve on both axes
        x_cursor = 0.0
        step_x, step_y = [], []

        for price, vol, label in entries:
            is_curt = label == "curt"
            is_dispatched = x_cursor < mismatch_volume

            if is_curt:
                color = color_curt
                hatch = "///"
                edgecolor = "black"
            elif is_dispatched:
                color = color_dispatched
                hatch = None
                edgecolor = "white"
            else:
                color = color_undispatched
                hatch = None
                edgecolor = "white"

            for ax in (ax_top, ax_bot):
                ax.bar(
                    x_cursor + vol / 2,
                    price,
                    width=vol,
                    color=color,
                    hatch=hatch,
                    edgecolor=edgecolor,
                    linewidth=0.6,
                    align="center",
                    zorder=2,
                )

            if vol > 10:
                lbl = "Curt." if is_curt else f"G{label}"
                target_ax = ax_top if is_curt else ax_bot
                target_ax.text(
                    x_cursor + vol / 2,
                    price + (self.C_curt * 0.01 if is_curt else 0.3),
                    lbl,
                    ha="center",
                    va="bottom",
                    fontsize=18,
                    color="black",
                )

            step_x += [x_cursor, x_cursor + vol]
            step_y += [price, price]
            x_cursor += vol

        #  Supply step curve (both axes)
        for ax in (ax_top, ax_bot):
            ax.step(
                step_x,
                step_y,
                where="pre",
                color="black",
                linewidth=2.5,
                linestyle="-",
                label="Upward bid curve",
                zorder=3,
            )

        # BM clearing price (both axes)
        for ax in (ax_top, ax_bot):
            ax.axhline(
                bm_price,
                color=color_bm_line,
                linewidth=2.5,
                linestyle="--",
                label=f"BM clearing price = ${bm_price:.2f}/MWh",
                zorder=4,
            )

        # Mismatch vertical line (both axes)
        for ax in (ax_top, ax_bot):
            ax.axvline(
                mismatch_volume,
                color=color_mismatch,
                linewidth=2.5,
                linestyle="--",
                label=f"Mismatch volume = {mismatch_volume:.1f} MW",
                zorder=4,
            )

        # Clearing point marker (bottom ax only)
        ax_bot.plot(
            mismatch_volume,
            bm_price,
            marker="x",
            markersize=14,
            markeredgewidth=2,
            color=color_bm_line,
            zorder=5,
            label=f"Clearing point ({mismatch_volume:.1f} MW, ${bm_price:.2f})",
        )

        #  Axis limits
        ax_bot.set_ylim(0, lower_max)
        ax_top.set_ylim(upper_min, upper_max)

        # Hide inner spines to create the visual break
        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        ax_top.tick_params(bottom=False)

        # Break marks
        d = 0.012
        kwargs = dict(
            transform=ax_top.transAxes, color="black", clip_on=False, linewidth=1.2
        )
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

        kwargs.update(transform=ax_bot.transAxes)
        ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        # Labels and formatting
        ax_bot.set_xlabel("Cumulative Upward Capacity (MW)", fontsize=21, labelpad=10)
        ax_bot.set_ylabel("Bid Price ($/MWh)", fontsize=21, labelpad=10)
        ax_bot.set_xlim(0, x_cursor * 1.02)
        ax_bot.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7, zorder=1)
        ax_top.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7, zorder=1)
        ax_top.tick_params(axis='both', labelsize=14)
        ax_bot.tick_params(axis='both', labelsize=14)


        # Legend (bottom ax only)
        legend_patches = [
            mpatches.Patch(
                facecolor=color_dispatched,
                edgecolor="white",
                label="Dispatched (upward regulation)",
            ),
            mpatches.Patch(
                facecolor=color_undispatched, edgecolor="white", label="Not dispatched"
            ),
            mpatches.Patch(
                facecolor=color_curt,
                hatch="///",
                edgecolor="black",
                label=f"Demand curtailment (${self.C_curt}/MWh)",
            ),
        ]
        handles, _ = ax_top.get_legend_handles_labels()
        ax_top.legend(
            handles=legend_patches + handles,
            fontsize=14,
            loc="upper left",
            framealpha=0.9,
        )

        plt.tight_layout()
        plt.savefig("results/step5_merit_order.png", dpi=150)

    def _plot_revenue_breakdown(self):
        """
        Plot a grouped bar chart comparing revenue breakdown under one-price vs two-price
        imbalance settlement schemes for all generating units.

        For each unit, two side-by-side bars are drawn:
            - Left bar:  one-price imbalance settlement
            - Right bar: two-price imbalance settlement

        Each bar is stacked with three components:
            - DAM profit:           day-ahead market profit (solid blue)
            - BSP revenue:          balancing service provider revenue (solid orange)
            - Imbalance settlement: positive (green hatched) or negative (red hatched)

        A diamond marker on each bar indicates total revenue (sum of all components).
        A vertical dashed line separates conventional generators from wind units.

        Output is saved to 'results/step5_revenue_breakdown.png'.
        """
        dam_profit = self.out_dict["day_ahead_market"]["profit_producers_DAM"]
        imb_one = self.out_dict["balancing_market"]["imbalance_settlements_one_price"]
        imb_two = self.out_dict["balancing_market"]["imbalance_settlements_two_price"]
        bsp_revenue = self.out_dict["balancing_market"]["revenue_BSPs"]

        units = list(dam_profit.keys())
        bsp_mapped = {f"gen_{j}": bsp_revenue.get(j, 0) for j in self.BSP}
        labels = [u.replace("gen_", "G").replace("wind_", "W") for u in units]

        def build_components(imb_dict):
            dam = [dam_profit.get(u, 0) for u in units]
            imb = [imb_dict.get(u, 0) for u in units]
            bsp = [bsp_mapped.get(u, 0) for u in units]
            total = [d + i + b for d, i, b in zip(dam, imb, bsp)]
            return dam, imb, bsp, total

        dam_one, imb_one_vals, bsp_vals, total_one = build_components(imb_one)
        dam_two, imb_two_vals, _, total_two = build_components(imb_two)

        color_dam = "#2196F3"
        color_bsp = "#FF9800"
        color_imb_pos = "#4CAF50"
        color_imb_neg = "#E53935"

        n = len(units)
        x = np.arange(n)
        # two bars per unit: left = one-price, right = two-price
        bar_w = 0.35
        off = 0.18  # offset from center

        fig, ax = plt.subplots(figsize=(16, 6))

        def draw_unit_bar(ax, idx, pos, dam, imb, bsp, total, is_one_price):
            lbl = is_one_price
            cursor = 0.0

            # 1. DAM profit — solid blue
            if abs(dam) > 1e-6:
                ax.bar(pos, dam, bar_w, bottom=cursor, color=color_dam, zorder=2)
            cursor += dam

            # 2. BSP revenue — solid orange (no blue underneath, no hatch needed)
            if bsp > 1e-6:
                ax.bar(
                    pos,
                    bsp,
                    bar_w,
                    bottom=cursor,
                    color=color_bsp,
                    zorder=2,
                    label="BSP revenue" if lbl and idx == 0 else "",
                )
                cursor += bsp

            # 3. Imbalance — blue base + hatched overlay so blue shows through
            if imb > 1e-6:
                ax.bar(pos, imb, bar_w, bottom=cursor, color=color_dam, zorder=2)
                ax.bar(
                    pos,
                    imb,
                    bar_w,
                    bottom=cursor,
                    facecolor=color_imb_pos,
                    hatch="///",
                    edgecolor=color_imb_pos,
                    alpha=0.6,
                    zorder=3,
                    label="Imbalance (+)" if lbl and idx == 0 else "",
                )
            elif imb < -1e-6:
                ax.bar(pos, imb, bar_w, bottom=cursor, color=color_dam, zorder=2)
                ax.bar(
                    pos,
                    imb,
                    bar_w,
                    bottom=cursor,
                    facecolor=color_imb_neg,
                    hatch="///",
                    edgecolor=color_imb_neg,
                    alpha=0.6,
                    zorder=3,
                    label="Imbalance (−)" if lbl and idx == 0 else "",
                )

            # 4. Total marker
            ax.plot(
                pos,
                total,
                marker="D",
                markersize=5,
                linestyle="none",
                color="black",
                zorder=5,
                label="Total revenue" if lbl and idx == 0 else "",
            )

        for idx in range(n):
            pos_one = x[idx] - off
            pos_two = x[idx] + off
            draw_unit_bar(
                ax,
                idx,
                pos_one,
                dam_one[idx],
                imb_one_vals[idx],
                bsp_vals[idx],
                total_one[idx],
                is_one_price=True,
            )
            draw_unit_bar(
                ax,
                idx,
                pos_two,
                dam_two[idx],
                imb_two_vals[idx],
                bsp_vals[idx],
                total_two[idx],
                is_one_price=False,
            )

        ax.axhline(0, color="black", linewidth=0.8, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Revenue ($)", fontsize=21, labelpad=10)
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7, zorder=1)

        gen_count = len(self.P_gens)
        ax.axvline(
            gen_count - 0.5, color="grey", linewidth=1.0, linestyle="--", zorder=3
        )

        # Legend
        legend_patches = [
            mpatches.Patch(
                facecolor=color_dam,
                edgecolor="white",
                label="DAM profit (blue base, both schemes)",
            ),
            mpatches.Patch(
                facecolor=color_bsp,
                hatch="///",
                edgecolor=color_bsp,
                alpha=0.6,
                label="BSP revenue",
            ),
            mpatches.Patch(
                facecolor=color_imb_pos,
                hatch="///",
                edgecolor=color_imb_pos,
                alpha=0.6,
                label="Imbalance settlement (+)",
            ),
            mpatches.Patch(
                facecolor=color_imb_neg,
                hatch="///",
                edgecolor=color_imb_neg,
                alpha=0.6,
                label="Imbalance settlement (−)",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="D",
                color="black",
                linestyle="none",
                markersize=5,
                label="Total revenue",
            ),
            mpatches.Patch(
                facecolor="white",
                edgecolor="grey",
                linestyle="--",
                label="← left bar: one-price  |  right bar: two-price →",
            ),
        ]
        ax.legend(handles=legend_patches, fontsize=8, loc="upper right", framealpha=0.9)

        plt.tight_layout()
        plt.savefig("results/step5_revenue_breakdown.png", dpi=150)
