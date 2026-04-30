import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

from src.input import InputHandler


class Model6:
    """
    Single-hour reserve capacity market clearing followed by single-hour day-ahead market clearing.
    """

    def __init__(
        self,
        hour: int,
        inp_hndl: InputHandler,
        up_balance_multiplier: float = 0.15,
        down_balance_multiplier: float = 0.1,
    ):
        """
        Runs and stores results for reserve and day-ahead market clearing at a given hour.

        On instantiation, solves the reserve and day-ahead clearing models sequentially.
        If both reach optimality, results are extracted and the merit order is plotted.

        Args:
            hour (int):              The hour of the day being simulated (0–23).
            inp_hndl (InputHandler): Object carrying all input data, including
                                     generators, wind farms, capacity factors,
                                     and demand profiles.

        Attributes:
            hour (int):              The hour of the day being simulated.
            P_gens (dict):           Conventional generator parameters.
            P_winds (dict):          Wind farm parameters.
            CF_wind (float):         Capacity factor for wind generation.
            Demands (dict):          Full demand profile dictionary.
            Load_sum (float):        Aggregated system load.
            reserve_model:           Solved Gurobi model for reserve clearing.
            day_ahead_model:         Solved Gurobi model for day-ahead clearing.
            out_dict (dict):         Parsed output from both markets. Only set
                                     if both models solve to optimality.
        """
        self.hour = hour
        self.P_gens = inp_hndl.generators
        self.P_winds = inp_hndl.wind_farms
        self.CF_wind = inp_hndl.CF_wind
        self.Demands = inp_hndl.demands
        self.Load_sum = inp_hndl.demands["system_load"]

        self.reserve_model = self._run_model_reserve_clearing(
            up_balance_multiplier, down_balance_multiplier
        )
        self.day_ahead_model = self._run_model_day_ahead_clearing()

        if (
            self.reserve_model.status == gp.GRB.OPTIMAL
            and self.day_ahead_model.status == gp.GRB.OPTIMAL
        ):
            self.out_dict = self._analyze_output()
            self._plot_merit_order()

    def _run_model_reserve_clearing(
        self, up_balance_multiplier: float = 0.15, down_balance_multiplier: float = 0.1,
        reserve_subset: set = None) -> gp.Model:
        """
        Build and solve the reserve capacity clearing model.

        Minimizes the total cost of procuring upward and downward reserve
        capacity from conventional generators, subject to:
            - Per-generator reserve capacity limits (R_plus, R_minus)
            - Per-generator combined reserve cap (P_max)
            - System-wide upward reserve requirement (15% of hourly load)
            - System-wide downward reserve requirement (10% of hourly load)

        Returns:
            gp.Model: Solved Gurobi model. Check `.status == GRB.OPTIMAL`
                      before extracting results.
        """
        model = gp.Model("model6")
        r_gen_up = {}
        r_gen_down = {}
        if reserve_subset is None:
            reserve_subset = {1, 2, 3, 4, 5, 6, 7}

        # Define Variables
        for i in reserve_subset:
            r_gen_up[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"r_gen_up_{i}")
            r_gen_down[i] = model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name=f"r_gen_down_{i}"
            )

        # Define Objective
        model.setObjective(
            gp.quicksum(self.P_gens[i]["C_up"] * r_gen_up[i] for i in reserve_subset)
            + gp.quicksum(
                self.P_gens[i]["C_down"] * r_gen_down[i] for i in reserve_subset
            ),
            GRB.MINIMIZE,
        )

        # Define Constraints
        # physical constraint conventional generators
        for i in reserve_subset:
            model.addConstr(
                r_gen_up[i] <= self.P_gens[i]["R_plus"], name=f"max_res_gen_up{i}"
            )
            model.addConstr(
                r_gen_down[i] <= self.P_gens[i]["R_minus"], name=f"max_res_gen_down{i}"
            )
            model.addConstr(
                r_gen_up[i] + r_gen_down[i] <= self.P_gens[i]["P_max"],
                name=f"max_res_gen_total{i}",
            )

        # power balance constraints
        model.addConstr(
            gp.quicksum(r_gen_up[i] for i in reserve_subset)
            == up_balance_multiplier * self.Load_sum[self.hour],
            name="reserve_balance_up",
        )
        model.addConstr(
            gp.quicksum(r_gen_down[i] for i in reserve_subset)
            == down_balance_multiplier * self.Load_sum[self.hour],
            name="reserve_balance_down",
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

    def _run_model_day_ahead_clearing(self) -> gp.Model:
        """
        Build and solve the day-ahead market clearing model.

        Maximizes social welfare (consumer surplus minus generation cost) by
        clearing generator and wind farm offers against flexible demand bids,
        subject to reserve capacity commitments from the reserve clearing model.

        Generators that cleared downward reserve capacity use a split-bid
        representation: a fixed block (forced minimum, bid at $0) plus a
        flexible block (bid at normal cost). This ensures the reserved
        downward capacity remains dispatchable in real time.

        Constraints:
            - Per-generator output bounded by [reserve_down, P_max - reserve_up]
            - Per-wind-farm output bounded by available capacity (P_max * CF)
            - Per-demand output bounded by load distribution share of system load
            - System-wide power balance (generation + wind == demand)

        Returns:
            gp.Model: Solved Gurobi model. Check `.status == GRB.OPTIMAL`
                      before extracting results.
        """
        model = gp.Model("model6_day_ahead")
        model.Params.InfUnbdInfo = 1
        p_gen = {}
        p_wind = {}
        demand = {}
        p_gen_flex = {}

        # extract cleared results from reserve model
        reserve_up = {
            i: (self.reserve_model.getVarByName(f"r_gen_up_{i}").X
                if self.reserve_model.getVarByName(f"r_gen_up_{i}") is not None
                else 0.0)
            for i in self.P_gens
        }
        reserve_down = {
            i: (self.reserve_model.getVarByName(f"r_gen_down_{i}").X
                if self.reserve_model.getVarByName(f"r_gen_down_{i}") is not None
                else 0.0)
            for i in self.P_gens
        }

        # Define Variables
        for i in self.P_gens:
            if reserve_down[i] > 1e-6:
                # Split bid: p_gen[i] is the forced minimum (bid at $0),
                # p_gen_flex[i] is the remainder bid at normal cost
                p_gen[i] = model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=reserve_down[i],
                    ub=reserve_down[i],
                    name=f"p_gen_{i}",
                )  # fixed at minimum
                p_gen_flex[i] = model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0,
                    ub=max(
                        0, self.P_gens[i]["P_max"] - reserve_up[i] - reserve_down[i]
                    ),
                    name=f"p_gen_flex_{i}",
                )
            else:
                p_gen[i] = model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0,
                    ub=max(0, self.P_gens[i]["P_max"] - reserve_up[i]),
                    name=f"p_gen_{i}",
                )
                p_gen_flex[i] = None

        # for i in self.P_gens:
        #     p_gen[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"p_gen_{i}")
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
            - gp.quicksum(
                self.P_gens[i]["C"]
                * (p_gen_flex[i] if p_gen_flex[i] is not None else p_gen[i])
                for i in self.P_gens
            ),
            GRB.MAXIMIZE,
        )
        # model.setObjective(gp.quicksum(demand[i]*self.Demands["bidding_prices"][i] for i in self.Demands["bidding_prices"]) - gp.quicksum(self.P_gens[i]["C"] * p_gen[i] for i in self.P_gens), GRB.MAXIMIZE)

        # Define Constraints
        # physical constraint conventional generators
        for i in self.P_gens:
            if reserve_down[i] > 1e-6:
                # Total output = forced minimum + flexible part
                model.addConstr(
                    p_gen[i] + p_gen_flex[i] <= self.P_gens[i]["P_max"] - reserve_up[i],
                    name=f"max_power_gen_{i}",
                )
                # min_power_gen is automatically satisfied since p_gen[i] is fixed at reserve_down[i]
            else:
                model.addConstr(
                    p_gen[i] <= self.P_gens[i]["P_max"] - reserve_up[i],
                    name=f"max_power_gen_{i}",
                )
                model.addConstr(p_gen[i] >= 0, name=f"min_power_gen_{i}")

        # for i in self.P_gens:
        #     model.addConstr(p_gen[i] <= (self.P_gens[i]["P_max"] - reserve_up[i]), name=f"max_power_gen_{i}")
        #     model.addConstr(p_gen[i] >= reserve_down[i], name=f"min_power_gen_{i}")
        # physical constraints wind farms
        for i in self.P_winds:
            model.addConstr(
                p_wind[i] <= self.P_winds[i]["P_max"] * self.CF_wind[self.hour],
                name=f"max_power_wind_{i}",
            )
        # physical constraints demands
        for i in self.Demands["load_distribution"]:
            model.addConstr(
                demand[i]
                <= self.Demands["load_distribution"][i] * self.Load_sum[self.hour],
                name=f"max_demand_load_{i}",
            )

        # power balance constraint
        model.addConstr(
            gp.quicksum(demand[i] for i in self.Demands["load_distribution"])
            - gp.quicksum(
                p_gen[i] + (p_gen_flex[i] if p_gen_flex[i] is not None else 0)
                for i in self.P_gens
            )
            - gp.quicksum(p_wind[i] for i in self.P_winds)
            == 0,
            name="power_balance",
        )

        # model.addConstr(gp.quicksum(demand[i] for i in self.Demands["load_distribution"])-gp.quicksum(p_gen[i] for i in self.P_gens)- gp.quicksum(p_wind[i] for i in self.P_winds) == 0, name=f"power_balance")

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
        Extract and aggregate results from the solved reserve and day-ahead models.

        Reads dual variables (market clearing prices) and primal variables
        (scheduled quantities) from both Gurobi models, then computes
        revenues, costs, and profits for all market participants.

        Returns:
            dict: Nested dictionary with three top-level keys:

                "reserve_market":
                    - market_clearing_price_up   (float): Dual of upward reserve balance.
                    - market_clearing_price_down (float): Dual of downward reserve balance.
                    - scheduled_upwards_regulation   (dict): Upward reserve per generator.
                    - scheduled_downwards_regulation (dict): Downward reserve per generator.
                    - revenue_BSPs (dict): Reserve revenue per generator.

                "day_ahead_market":
                    - market_clearing_price_DAM (float): Dual of power balance constraint.
                    - social_welfare            (float): Optimal objective value.
                    - total_operating_cost      (float): Total generation cost.
                    - scheduled_power_output    (dict): Cleared output per unit.
                    - DAM_revenue               (dict): Revenue per unit at DAM price.
                    - DAM_cost                  (dict): Generation cost per unit.
                    - utility_demands           (list): Consumer surplus per demand.

                "total_results":
                    - total_revenues (dict): DAM + reserve revenue per unit.
                    - total_profits  (dict): Total revenue minus generation cost per unit.
        """
        # Reserve Market (RM)
        reserve_price_up = self.reserve_model.getConstrByName("reserve_balance_up").Pi
        reserve_price_down = self.reserve_model.getConstrByName("reserve_balance_down").Pi

        def _get_reserve_val(var_name):
            var = self.reserve_model.getVarByName(var_name)
            return var.X if var is not None else 0.0

        reserve_up_schedule = {
            f"gen_{i}": _get_reserve_val(f"r_gen_up_{i}")
            for i in self.P_gens
        }
        reserve_down_schedule = {
            f"gen_{i}": _get_reserve_val(f"r_gen_down_{i}")
            for i in self.P_gens
        }

        reserve_revenue = {
            i: reserve_up_schedule[i] * reserve_price_up
            + reserve_down_schedule[i] * reserve_price_down * -1
            for i in reserve_up_schedule
        }
        #Reserve Market clearing
        reserve_cost = self.reserve_model.ObjVal
        # Day-Ahead Market (DAM)
        dam_clearing = self.day_ahead_model.getConstrByName("power_balance").Pi
        social_welfare = self.day_ahead_model.ObjVal

        #Total SW
        tot_social_welfare = social_welfare-reserve_cost
        # reserve_down_vals = {
        #     i: self.reserve_model.getVarByName(f"r_gen_down_{i}").X for i in self.P_gens
        # }

        dam_schedule = {}
        for i in self.P_gens:
            base = self.day_ahead_model.getVarByName(f"p_gen_{i}").X
            flex_var = self.day_ahead_model.getVarByName(f"p_gen_flex_{i}")
            flex = flex_var.X if flex_var is not None else 0.0
            dam_schedule[f"gen_{i}"] = base + flex

        dam_schedule |= {
            f"wind_{i}": self.day_ahead_model.getVarByName(f"p_wind_{i}").X
            for i in self.P_winds
        }

        revenue_producers_DAM = {
            f"gen_{i}": dam_schedule[f"gen_{i}"] * dam_clearing for i in self.P_gens
        } | {
            f"wind_{i}": self.day_ahead_model.getVarByName(f"p_wind_{i}").X
            * dam_clearing
            for i in self.P_winds
        }

        cost_producers_DAM = {
            f"gen_{i}": dam_schedule[f"gen_{i}"] * self.P_gens[i]["C"]
            for i in self.P_gens
        }

        # dam_schedule = {
        #     f"gen_{i}": self.day_ahead_model.getVarByName(f"p_gen_{i}").X
        #     for i in self.P_gens
        # } | {
        #     f"wind_{i}": self.day_ahead_model.getVarByName(f"p_wind_{i}").X
        #     for i in self.P_winds
        # }

        # revenue_producers_DAM = {
        #     f"gen_{i}": self.day_ahead_model.getVarByName(f"p_gen_{i}").X * (dam_clearing)
        #     for i in self.P_gens
        # } | {
        #     f"wind_{i}": self.day_ahead_model.getVarByName(f"p_wind_{i}").X * dam_clearing
        #     for i in self.P_winds
        # }

        # cost_producers_DAM = {
        #     f"gen_{i}": self.day_ahead_model.getVarByName(f"p_gen_{i}").X * self.P_gens[i]["C"]
        #     for i in self.P_gens
        # }

        utility_demands = [
            self.day_ahead_model.getVarByName(f"demand_{i}").X
            * (self.Demands["bidding_prices"][i] - dam_clearing)
            for i in self.Demands["load_distribution"]
        ]

        total_operating_cost = sum(
            self.day_ahead_model.getVarByName(f"p_gen_{i}").X * self.P_gens[i]["C"]
            for i in self.P_gens
        )

        out_dict = {}
        out_dict["reserve_market"] = {
            "solve_start_perf_counter": getattr(self.reserve_model, "_solve_start_perf", None),
            "solve_end_perf_counter": getattr(self.reserve_model, "_solve_end_perf", None),
            "solve_time_seconds": getattr(self.reserve_model, "_solve_time_seconds", None),
            "market_clearing_price_up": reserve_price_up,
            "market_clearing_price_down": reserve_price_down,
            "scheduled_upwards_regulation": reserve_up_schedule,
            "scheduled_downwards_regulation": reserve_down_schedule,
            "revenue_BSPs": reserve_revenue,
            "reserve_cost": reserve_cost
        }

        out_dict["day_ahead_market"] = {
            "solve_start_perf_counter": getattr(self.day_ahead_model, "_solve_start_perf", None),
            "solve_end_perf_counter": getattr(self.day_ahead_model, "_solve_end_perf", None),
            "solve_time_seconds": getattr(self.day_ahead_model, "_solve_time_seconds", None),
            "market_clearing_price_DAM": dam_clearing,
            "social_welfare": social_welfare,
            "total_operating_cost": total_operating_cost,
            "scheduled_power_output": dam_schedule,
            "DAM_revenue": revenue_producers_DAM,
            "DAM_cost": cost_producers_DAM,
            "utility_demands": utility_demands,
        }

        out_dict["total_results"] = {
            "total_revenues": {
                i: revenue_producers_DAM.get(i, 0.0) + reserve_revenue.get(i, 0.0)
                for i in set(
                    list(revenue_producers_DAM.keys()) + list(reserve_revenue.keys())
                )
            },
            "total_profits": {
                i: revenue_producers_DAM.get(i, 0.0)
                + reserve_revenue.get(i, 0.0)
                - cost_producers_DAM.get(i, 0.0)
                for i in set(
                    list(revenue_producers_DAM.keys()) + list(reserve_revenue.keys())
                )
            },
            "Total Social Welfare": tot_social_welfare
        }
        return out_dict

    def _plot_merit_order(self):
        """
        Plot the day-ahead market merit order curve with supply and demand bids.

        Builds the supply stack in merit order, accounting for split-bid generators
        (those with cleared downward reserve capacity appear as two blocks: a forced
        minimum at $0 and a flexible remainder at normal cost). Wind capacity is
        prepended at zero cost. Each bar is color-coded by type:

            - Blue  (hatched): Wind generation
            - Purple (hatched): Forced reserve minimum block (bid at $0)
            - Cyan  (hatched): Zero-cost conventional block
            - Green:           Dispatched flexible capacity
            - Grey:            Undispatched flexible capacity

        The demand curve is plotted in descending price order. The intersection
        of supply and demand is marked with the market clearing price (MCP).

        Output is saved to 'results/step6_merit_order.png'.
        """
        mcp = self.out_dict["day_ahead_market"]["market_clearing_price_DAM"]

        # --- Retrieve reserve values ---
        def _get_reserve_val(var_name):
            var = self.reserve_model.getVarByName(var_name)
            return var.X if var is not None else 0.0

        reserve_up = {
            i: _get_reserve_val(f"r_gen_up_{i}") for i in self.P_gens
        }
        reserve_down = {
            i: _get_reserve_val(f"r_gen_down_{i}") for i in self.P_gens
        }

        # --- Build generator stack with split bids ---
        # Each generator may produce up to 2 entries:
        #   (a) forced minimum block at $0 (only if reserve_down > 0)
        #   (b) flexible block at normal cost
        raw_entries = []  # list of (cost, cap, label, is_forced)
        for i in self.P_gens:
            flex_cap = self.P_gens[i]["P_max"] - reserve_up[i] - reserve_down[i]
            if reserve_down[i] > 1e-6:
                raw_entries.append((0.0, reserve_down[i], i, True))
            if flex_cap > 1e-6:
                raw_entries.append((self.P_gens[i]["C"], flex_cap, i, False))

        # Sort all entries by cost (merit order), forced $0 blocks naturally come first
        raw_entries.sort(key=lambda x: x[0])

        # --- Wind block (zero cost, prepend before any $0 gen blocks) ---
        wind_caps = [
            self.P_winds[i]["P_max"] * self.CF_wind[self.hour] for i in self.P_winds
        ]
        total_wind = sum(wind_caps)

        # Final ordered lists for supply curve
        all_costs = [0.0] + [e[0] for e in raw_entries]
        all_caps = [total_wind] + [e[1] for e in raw_entries]
        all_labels = ["Wind"] + [e[2] for e in raw_entries]
        all_forced = [False] + [e[3] for e in raw_entries]

        # --- Dispatched set (total output per generator > 0) ---
        dispatched_set = set()
        scheduled = self.out_dict["day_ahead_market"]["scheduled_power_output"]
        for key in self.P_gens.keys():
            if scheduled.get(f"gen_{key}", 0.0) > 1e-6:
                dispatched_set.add(key)
        if any(scheduled.get(f"wind_{key}", 0.0) > 1e-6 for key in self.P_winds.keys()):
            dispatched_set.add("Wind")

        # --- Total demand ---
        total_demand = self.Load_sum[self.hour]

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(14, 6))

        COLOR_WIND = "#2196F3"  # blue
        COLOR_DISPATCHED = "#4CAF50"  # green
        COLOR_FORCED = "#9C27B0"  # purple
        COLOR_FREE = "#00BCD4"  # cyan
        COLOR_UNDISPATCHED = "#BDBDBD"  # grey
        COLOR_MCP_LINE = "#E53935"
        COLOR_DEMAND_LINE = "#FF6F00"

        # Height to give $0 bars so they are visible
        ZERO_BAR_HEIGHT = 0.8

        x_cursor = 0.0
        for cap, cost, label, is_forced in zip(
            all_caps, all_costs, all_labels, all_forced
        ):
            if cap < 1e-6:
                continue
            is_wind = label == "Wind"

            if is_wind:
                color = COLOR_WIND
                hatch = "///"
                height = ZERO_BAR_HEIGHT
            elif is_forced:
                color = COLOR_FORCED
                hatch = "xxx"
                height = ZERO_BAR_HEIGHT
            elif cost == 0.0:
                color = COLOR_FREE
                hatch = "..."
                height = ZERO_BAR_HEIGHT
            else:
                total_scheduled = scheduled.get(f"gen_{label}", 0.0)
                forced_min = reserve_down.get(label, 0.0)
                flex_dispatched = total_scheduled - forced_min
                color = (
                    COLOR_DISPATCHED if flex_dispatched > 1e-6 else COLOR_UNDISPATCHED
                )
                hatch = None
                height = cost

            ax.bar(
                x_cursor + cap / 2,
                height,
                width=cap,
                color=color,
                hatch=hatch,
                edgecolor="white" if hatch is None else "black",
                linewidth=0.6,
                align="center",
                zorder=2,
            )

            if cap > 15:
                if is_wind:
                    display_label = "Wind 1-6"
                elif is_forced:
                    display_label = f"G{label}*"
                else:
                    display_label = f"G{label}"
                ax.text(
                    x_cursor + cap / 2,
                    height + 0.15,
                    display_label,
                    ha="center",
                    va="bottom",
                    fontsize=16,
                    color="black",
                )
            x_cursor += cap

        total_capacity = x_cursor

        # --- Step line (supply merit order curve) ---
        step_x, step_y, x_cursor2 = [], [], 0.0
        for cap, cost in zip(all_caps, all_costs):
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
            label="Supply bid curve",
            zorder=3,
        )

        # --- MCP line ---
        ax.axhline(
            mcp,
            color=COLOR_MCP_LINE,
            linewidth=2.5,
            linestyle="--",
            label=f"Market clearing price = ${mcp:.2f}/MWh",
            zorder=4,
        )

        # --- Demand vertical line ---
        ax.axvline(
            total_demand,
            color=COLOR_DEMAND_LINE,
            linewidth=2.5,
            linestyle="--",
            label=f"Total demand = {total_demand:.1f} MW",
            zorder=4,
        )

        # --- Demand merit order curve ---
        demand_keys = list(self.Demands["load_distribution"].keys())
        demand_quantities = [
            self.Demands["load_distribution"][i] * self.Load_sum[self.hour]
            for i in demand_keys
        ]
        demand_prices = [self.Demands["bidding_prices"][i] for i in demand_keys]

        sorted_demands = sorted(
            zip(demand_prices, demand_quantities), key=lambda x: x[0], reverse=True
        )
        sorted_d_prices, sorted_d_quantities = zip(*sorted_demands)

        demand_step_x, demand_step_y, x_cursor_d = [], [], 0.0
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
            color=COLOR_DEMAND_LINE,
            linewidth=2.5,
            linestyle="-",
            label="Demand bid curve",
            zorder=3,
        )

        # --- Clearing point marker ---
        ax.plot(
            total_demand,
            mcp,
            marker="o",
            markersize=14,
            color=COLOR_MCP_LINE,
            zorder=5,
            label=f"Clearing point ({total_demand:.1f} MW, ${mcp:.2f})",
        )

        # --- Formatting ---
        ax.set_xlabel("Cumulative Capacity (MW)", fontsize=21)
        ax.set_ylabel("Bid Price ($/MWh)", fontsize=21)
        ax.set_xlim(0, max(total_capacity, x_cursor_d) * 1.02)
        ax.set_ylim(-0.5, max(all_costs) * 2 + 1)
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7, zorder=1)
        ax.tick_params(axis='both', labelsize=15)

        # --- Legend ---
        legend_patches = [
            mpatches.Patch(
                facecolor=COLOR_WIND,
                hatch="///",
                edgecolor="black",
                label="Wind (zero marginal cost)",
            ),
            mpatches.Patch(
                facecolor=COLOR_FORCED,
                hatch="xxx",
                edgecolor="black",
                label="Down reserve (zero cost forced)",
            ),
            mpatches.Patch(
                facecolor=COLOR_FREE,
                hatch="...",
                edgecolor="black",
                label="Zero-cost conventional",
            ),
            mpatches.Patch(
                facecolor=COLOR_DISPATCHED,
                edgecolor="white",
                label="Dispatched generators",
            ),
            mpatches.Patch(
                facecolor=COLOR_UNDISPATCHED, edgecolor="white", label="Not dispatched"
            ),
        ]
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(
            handles=legend_patches + handles,
            fontsize=10.5,
            loc="upper right",
            framealpha=0.9,
        )

        plt.tight_layout()
        plt.savefig("results/step6_merit_order.png", dpi=150)
