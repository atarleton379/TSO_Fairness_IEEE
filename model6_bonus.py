import gurobipy as gp
from gurobipy import GRB
from src.input import InputHandler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import time

class Model6Bonus:
    def __init__(self, HOUR: int, INP_HNDL: InputHandler):
        self.hour = HOUR
        # self.inp_hndl = INP_HNDL
        self.P_gens = INP_HNDL.generators
        self.P_winds = INP_HNDL.wind_farms
        self.CF_wind = INP_HNDL.CF_wind
        self.Demands = INP_HNDL.demands
        self.Load_sum = INP_HNDL.demands["system_load"]

        self.out_dict = {}

        self.combined_model = self._run_model()

        if self.combined_model.status == gp.GRB.OPTIMAL and self.combined_model.status == gp.GRB.OPTIMAL:
            self.out_dict = self._analyze_output()
            # self._plot_merit_order()

    def _run_model(self) -> gp.Model:
        model = gp.Model("model6")
        r_gen_up = {}
        r_gen_down = {}
        p_gen = {}
        p_wind = {}
        demand = {}
         


        # Define Variables
        for i in self.P_gens:
            r_gen_up[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"r_gen_up_{i}")
            r_gen_down[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"r_gen_down_{i}")
            p_gen[i]      = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub = self.P_gens[i]["P_max"], name=f"p_gen_{i}")
        
        for i in self.P_winds:
            p_wind[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"p_wind_{i}")
        for i in self.Demands["load_distribution"]:
                demand[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"demand_{i}")   
        
        # Define Objective      
        model.setObjective(gp.quicksum(demand[i]*self.Demands["bidding_prices"][i] for i in self.Demands["bidding_prices"]) # Demands 
                           - gp.quicksum(p_gen[i] *self.P_gens[i]["C"] for i in self.P_gens) # Generator Costs
                            -(gp.quicksum(self.P_gens[i]["C_up"] * r_gen_up[i] for i in self.P_gens) #negated upregulation costs
                            + gp.quicksum(self.P_gens[i]["C_down"] * r_gen_down[i] for i in self.P_gens)), #negated down regulation costs
                            GRB.MAXIMIZE)

        ## Define Constraints
        #physical constraint conventional generators
        for i in self.P_gens:
            model.addConstr(p_gen[i] <= (self.P_gens[i]["P_max"] - r_gen_up[i]), name=f"max_power_gen_{i}")
            model.addConstr(p_gen[i] >= r_gen_down[i], name=f"min_power_gen_{i}")

            #regulation limits
            model.addConstr(r_gen_up[i] <= self.P_gens[i]["R_plus"], name=f"max_res_gen_up{i}")
            model.addConstr(r_gen_down[i] <= self.P_gens[i]["R_minus"], name=f"max_res_gen_down{i}") 
            model.addConstr(r_gen_up[i] + r_gen_down[i] <= self.P_gens[i]["P_max"], name=f"max_res_gen_total{i}")

        #physical constraints wind farms    
        for i in self.P_winds:
            model.addConstr(p_wind[i] <= self.P_winds[i]["P_max"]*self.CF_wind[self.hour], name=f"max_power_wind_{i}") 
        # physical constraints demands
        for i in self.Demands["load_distribution"]:
            model.addConstr(demand[i] <= self.Demands["load_distribution"][i]*self.Load_sum[self.hour], name=f"max_demand_load_{i}") 

        #balance constraint
        model.addConstr(
        gp.quicksum(demand[i] for i in self.Demands["load_distribution"])
        - gp.quicksum(
            p_gen[i]
            for i in self.P_gens
        )
        - gp.quicksum(p_wind[i] for i in self.P_winds) == 0,
        name="power_balance"
        )


        #balance constraints
        model.addConstr(gp.quicksum(r_gen_up[i]for i in self.P_gens) 
                        == 0.15*self.Load_sum[self.hour], name=f"reserve_balance_up")
        model.addConstr(gp.quicksum(r_gen_down[i]for i in self.P_gens) 
                        == 0.10*self.Load_sum[self.hour], name=f"reserve_balance_down")

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
    
    #     p_gen = {}
    #     p_wind = {}
    #     demand = {}
    #     p_gen_flex = {} 

    #     reserve_up = {i: self.combined_model.getVarByName(f"r_gen_up_{i}").X for i in self.P_gens}
    #     reserve_down = {i: self.combined_model.getVarByName(f"r_gen_down_{i}").X for i in self.P_gens}
    
    #     # Define Variables
    #     for i in self.P_gens:
    #         if reserve_down[i] > 1e-6:
    #             # Split bid: p_gen[i] is the forced minimum (bid at $0),
    #             # p_gen_flex[i] is the remainder bid at normal cost
    #             p_gen[i]      = model.addVar(vtype=GRB.CONTINUOUS, lb=reserve_down[i],
    #                                         ub=reserve_down[i], name=f"p_gen_{i}")  # fixed at minimum
    #             p_gen_flex[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
    #                                         ub=max(0, self.P_gens[i]["P_max"] - reserve_up[i] - reserve_down[i]),
    #                                         name=f"p_gen_flex_{i}")
    #         else:
    #             p_gen[i]      = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
    #                                         ub=max(0, self.P_gens[i]["P_max"] - reserve_up[i]),
    #                                         name=f"p_gen_{i}")
    #             p_gen_flex[i] = None
        
    #     # for i in self.P_gens:
    #     #     p_gen[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"p_gen_{i}")
    #     for i in self.P_winds:
    #         p_wind[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"p_wind_{i}")
    #     for i in self.Demands["load_distribution"]:
    #             demand[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"demand_{i}")

    #     # Define Objective

    #     # model.setObjective(gp.quicksum(demand[i]*self.Demands["bidding_prices"][i] for i in self.Demands["bidding_prices"]) - gp.quicksum(self.P_gens[i]["C"] * p_gen[i] for i in self.P_gens), GRB.MAXIMIZE)


    #     ## Define Constraints
    #     #physical constraint conventional generators
    #     for i in self.P_gens:
    #         if reserve_down[i] > 1e-6:
    #             # Total output = forced minimum + flexible part
    #             model.addConstr(
    #                 p_gen[i] + p_gen_flex[i] <= self.P_gens[i]["P_max"] - reserve_up[i],
    #                 name=f"max_power_gen_{i}"
    #             )
    #             # min_power_gen is automatically satisfied since p_gen[i] is fixed at reserve_down[i]
    #         else:
    #             model.addConstr(p_gen[i] <= self.P_gens[i]["P_max"] - reserve_up[i],
    #                             name=f"max_power_gen_{i}")
    #             model.addConstr(p_gen[i] >= 0, name=f"min_power_gen_{i}")

    #     # for i in self.P_gens:
    #     #     model.addConstr(p_gen[i] <= (self.P_gens[i]["P_max"] - reserve_up[i]), name=f"max_power_gen_{i}")
    #     #     model.addConstr(p_gen[i] >= reserve_down[i], name=f"min_power_gen_{i}")
    #     #physical constraints wind farms    
    #     for i in self.P_winds:
    #         model.addConstr(p_wind[i] <= self.P_winds[i]["P_max"]*self.CF_wind[self.hour], name=f"max_power_wind_{i}") 
    #     # physical constraints demands
    #     for i in self.Demands["load_distribution"]:
    #         model.addConstr(demand[i] <= self.Demands["load_distribution"][i]*self.Load_sum[self.hour], name=f"max_demand_load_{i}") 

    #     #balance constraint
    #     model.addConstr(
    #     gp.quicksum(demand[i] for i in self.Demands["load_distribution"])
    #     - gp.quicksum(
    #         p_gen[i] + (p_gen_flex[i] if p_gen_flex[i] is not None else 0)
    #         for i in self.P_gens
    #     )
    #     - gp.quicksum(p_wind[i] for i in self.P_winds) == 0,
    #     name="power_balance"
    # )    

    #     # model.addConstr(gp.quicksum(demand[i] for i in self.Demands["load_distribution"])-gp.quicksum(p_gen[i] for i in self.P_gens)- gp.quicksum(p_wind[i] for i in self.P_winds) == 0, name=f"power_balance")

    #     # Optimize the model
    #     model.optimize()

        # if model.status == gp.GRB.OPTIMAL:
        #     print("Optimal solution found!")
        # else:
        #     print(f"Solver status: {model.status}")
        # return model
    
    def _analyze_output(self) -> dict:
        # Reserve Market (RM)
        reserve_price_up = self.combined_model.getConstrByName("reserve_balance_up").Pi *-1
        reserve_price_down = self.combined_model.getConstrByName("reserve_balance_down").Pi *-1

        reserve_up_schedule = {
            f"gen_{i}": self.combined_model.getVarByName(f"r_gen_up_{i}").X
            for i in self.P_gens
        }
        reserve_down_schedule = {
            f"gen_{i}": self.combined_model.getVarByName(f"r_gen_down_{i}").X
            for i in self.P_gens
        }

        reserve_revenue = {
            i: reserve_up_schedule[i] * reserve_price_up + reserve_down_schedule[i] * reserve_price_down *-1
            for i in reserve_up_schedule
        }


        # Day-Ahead Market (DAM)
        dam_clearing = self.combined_model.getConstrByName("power_balance").Pi
        social_welfare = self.combined_model.ObjVal

        reserve_down_vals = {i: self.combined_model.getVarByName(f"r_gen_down_{i}").X for i in self.P_gens}

        dam_schedule = {}
        for i in self.P_gens:
            base = self.combined_model.getVarByName(f"p_gen_{i}").X
            dam_schedule[f"gen_{i}"] = base

        dam_schedule |= {
            f"wind_{i}": self.combined_model.getVarByName(f"p_wind_{i}").X
            for i in self.P_winds
        }

        revenue_producers_DAM = {
            f"gen_{i}": dam_schedule[f"gen_{i}"] * dam_clearing
            for i in self.P_gens
        } | {
            f"wind_{i}": self.combined_model.getVarByName(f"p_wind_{i}").X * dam_clearing
            for i in self.P_winds
        }

        cost_producers_DAM = {
            f"gen_{i}": dam_schedule[f"gen_{i}"] * self.P_gens[i]["C"]
            for i in self.P_gens
        }

        # dam_schedule = {
        #     f"gen_{i}": self.combined_model.getVarByName(f"p_gen_{i}").X
        #     for i in self.P_gens
        # } | {
        #     f"wind_{i}": self.combined_model.getVarByName(f"p_wind_{i}").X
        #     for i in self.P_winds
        # }

        # revenue_producers_DAM = {
        #     f"gen_{i}": self.combined_model.getVarByName(f"p_gen_{i}").X * (dam_clearing)
        #     for i in self.P_gens
        # } | {
        #     f"wind_{i}": self.combined_model.getVarByName(f"p_wind_{i}").X * dam_clearing
        #     for i in self.P_winds
        # }

        # cost_producers_DAM = {
        #     f"gen_{i}": self.combined_model.getVarByName(f"p_gen_{i}").X * self.P_gens[i]["C"]
        #     for i in self.P_gens
        # }

        utility_demands = [
            self.combined_model.getVarByName(f"demand_{i}").X * (self.Demands["bidding_prices"][i] - dam_clearing)
            for i in self.Demands["load_distribution"]
        ]

        total_operating_cost = sum(
            self.combined_model.getVarByName(f"p_gen_{i}").X * self.P_gens[i]["C"]
            for i in self.P_gens
        )


        out_dict = {}
        out_dict["reserve_market"] = {
            "solve_start_perf_counter": getattr(self.combined_model, "_solve_start_perf", None),
            "solve_end_perf_counter": getattr(self.combined_model, "_solve_end_perf", None),
            "solve_time_seconds": getattr(self.combined_model, "_solve_time_seconds", None),
            "market_clearing_price_up": reserve_price_up,
            "market_clearing_price_down": reserve_price_down,
            "scheduled_upwards_regulation": reserve_up_schedule,
            "scheduled_downwards_regulation": reserve_down_schedule,
            "revenue_BSPs": reserve_revenue,
        }
        
        out_dict["day_ahead_market"] = {
            "solve_start_perf_counter": getattr(self.combined_model, "_solve_start_perf", None),
            "solve_end_perf_counter": getattr(self.combined_model, "_solve_end_perf", None),
            "solve_time_seconds": getattr(self.combined_model, "_solve_time_seconds", None),
            "market_clearing_price_DAM": dam_clearing,
            "social_welfare": social_welfare,
            "total_operating_cost": total_operating_cost,
            "scheduled_power_output": dam_schedule,
            "DAM_revenue":revenue_producers_DAM,
            "DAM_cost": cost_producers_DAM,
            "utility_demands": utility_demands,
        }

        out_dict["total_results"] = {
            "total_revenues": {
                i: revenue_producers_DAM.get(i, 0.0) + reserve_revenue.get(i, 0.0)
                for i in set(list(revenue_producers_DAM.keys()) + list(reserve_revenue.keys()))
            },
            "total_profits": {
                i: revenue_producers_DAM.get(i, 0.0) + reserve_revenue.get(i, 0.0) - cost_producers_DAM.get(i, 0.0)
                for i in set(list(revenue_producers_DAM.keys()) + list(reserve_revenue.keys()))
            }   
        }

        return out_dict

    # def _plot_merit_order(self):
    #     mcp = self.out_dict["day_ahead_market"]["market_clearing_price_DAM"]

    #     # --- Retrieve reserve values ---
    #     reserve_up   = {i: self.combined_model.getVarByName(f"r_gen_up_{i}").X   for i in self.P_gens}
    #     reserve_down = {i: self.combined_model.getVarByName(f"r_gen_down_{i}").X for i in self.P_gens}

    #     # --- Build generator stack with split bids ---
    #     # Each generator may produce up to 2 entries:
    #     #   (a) forced minimum block at $0 (only if reserve_down > 0)
    #     #   (b) flexible block at normal cost
    #     raw_entries = []  # list of (cost, cap, label, is_forced)
    #     for i in self.P_gens:
    #         flex_cap = self.P_gens[i]["P_max"] - reserve_up[i] - reserve_down[i]
    #         if reserve_down[i] > 1e-6:
    #             raw_entries.append((0.0,                  reserve_down[i], i, True))
    #         if flex_cap > 1e-6:
    #             raw_entries.append((self.P_gens[i]["C"],  flex_cap,        i, False))

    #     # Sort all entries by cost (merit order), forced $0 blocks naturally come first
    #     raw_entries.sort(key=lambda x: x[0])

    #     # --- Wind block (zero cost, prepend before any $0 gen blocks) ---
    #     wind_caps  = [self.P_winds[i]["P_max"] * self.CF_wind[self.hour] for i in self.P_winds]
    #     total_wind = sum(wind_caps)

    #     # Final ordered lists for supply curve
    #     all_costs  = [0.0]    + [e[0] for e in raw_entries]
    #     all_caps   = [total_wind] + [e[1] for e in raw_entries]
    #     all_labels = ["Wind"] + [e[2] for e in raw_entries]
    #     all_forced = [False]  + [e[3] for e in raw_entries]

    #     # --- Dispatched set (total output per generator > 0) ---
    #     dispatched_set = set()
    #     scheduled = self.out_dict["day_ahead_market"]["scheduled_power_output"]
    #     for key in self.P_gens.keys():
    #         if scheduled.get(f"gen_{key}", 0.0) > 1e-6:
    #             dispatched_set.add(key)
    #     if any(scheduled.get(f"wind_{key}", 0.0) > 1e-6 for key in self.P_winds.keys()):
    #         dispatched_set.add("Wind")

    #     # --- Total demand ---
    #     total_demand = self.Load_sum[self.hour]

    #     # --- Plot ---
    #     fig, ax = plt.subplots(figsize=(14, 6))

    #     COLOR_WIND         = "#2196F3"  # blue
    #     COLOR_DISPATCHED   = "#4CAF50"  # green
    #     COLOR_FORCED       = "#9C27B0"  # purple
    #     COLOR_FREE         = "#00BCD4"  # cyan
    #     COLOR_UNDISPATCHED = "#BDBDBD"  # grey
    #     COLOR_MCP_LINE     = "#E53935"
    #     COLOR_DEMAND_LINE  = "#FF6F00"

    #     # Height to give $0 bars so they are visible
    #     ZERO_BAR_HEIGHT = 0.8

    #     x_cursor = 0.0
    #     for cap, cost, label, is_forced in zip(all_caps, all_costs, all_labels, all_forced):
    #         if cap < 1e-6:
    #             continue
    #         is_wind = (label == "Wind")

    #         if is_wind:
    #             color   = COLOR_WIND
    #             hatch   = "///"
    #             height  = ZERO_BAR_HEIGHT
    #         elif is_forced:
    #             color   = COLOR_FORCED
    #             hatch   = "xxx"
    #             height  = ZERO_BAR_HEIGHT
    #         elif cost == 0.0:
    #             color   = COLOR_FREE
    #             hatch   = "..."
    #             height  = ZERO_BAR_HEIGHT
    #         else:
    #             total_scheduled = scheduled.get(f"gen_{label}", 0.0)
    #             forced_min      = reserve_down.get(label, 0.0)
    #             flex_dispatched = total_scheduled - forced_min
    #             color   = COLOR_DISPATCHED if flex_dispatched > 1e-6 else COLOR_UNDISPATCHED
    #             hatch   = None
    #             height  = cost

    #         ax.bar(
    #             x_cursor + cap / 2, height,
    #             width=cap,
    #             color=color,
    #             hatch=hatch,
    #             edgecolor="white" if hatch is None else "black",
    #             linewidth=0.6,
    #             align="center",
    #             zorder=2
    #         )

    #         if cap > 15:
    #             if is_wind:
    #                 display_label = "Wind 1-6"
    #             elif is_forced:
    #                 display_label = f"G{label}*"
    #             else:
    #                 display_label = f"G{label}"
    #             ax.text(
    #                 x_cursor + cap / 2, height + 0.15,
    #                 display_label,
    #                 ha="center", va="bottom", fontsize=7.5, color="black"
    #             )
    #         x_cursor += cap

    #     total_capacity = x_cursor

    #     # --- Step line (supply merit order curve) ---
    #     step_x, step_y, x_cursor2 = [], [], 0.0
    #     for cap, cost in zip(all_caps, all_costs):
    #         if cap < 1e-6:
    #             continue
    #         step_x.append(x_cursor2)
    #         step_y.append(cost)
    #         x_cursor2 += cap
    #         step_x.append(x_cursor2)
    #         step_y.append(cost)

    #     ax.step(step_x, step_y, where="pre", color="black",
    #             linewidth=1.4, linestyle="-", label="Supply bid curve", zorder=3)

    #     # --- MCP line ---
    #     ax.axhline(mcp, color=COLOR_MCP_LINE, linewidth=1.8, linestyle="--",
    #             label=f"Market clearing price = ${mcp:.2f}/MWh", zorder=4)

    #     # --- Demand vertical line ---
    #     ax.axvline(total_demand, color=COLOR_DEMAND_LINE, linewidth=1.8, linestyle="--",
    #             label=f"Total demand = {total_demand:.1f} MW", zorder=4)

    #     # --- Demand merit order curve ---
    #     demand_keys = list(self.Demands["load_distribution"].keys())
    #     demand_quantities = [self.Demands["load_distribution"][i] * self.Load_sum[self.hour]
    #                         for i in demand_keys]
    #     demand_prices = [self.Demands["bidding_prices"][i] for i in demand_keys]

    #     sorted_demands = sorted(zip(demand_prices, demand_quantities),
    #                             key=lambda x: x[0], reverse=True)
    #     sorted_d_prices, sorted_d_quantities = zip(*sorted_demands)

    #     demand_step_x, demand_step_y, x_cursor_d = [], [], 0.0
    #     for qty, price in zip(sorted_d_quantities, sorted_d_prices):
    #         if qty < 1e-6:
    #             continue
    #         demand_step_x.append(x_cursor_d)
    #         demand_step_y.append(price)
    #         x_cursor_d += qty
    #         demand_step_x.append(x_cursor_d)
    #         demand_step_y.append(price)

    #     ax.step(demand_step_x, demand_step_y, where="pre",
    #             color=COLOR_DEMAND_LINE, linewidth=1.8, linestyle="-",
    #             label="Demand bid curve", zorder=3)

    #     # --- Clearing point marker ---
    #     ax.plot(total_demand, mcp, marker="o", markersize=9,
    #             color=COLOR_MCP_LINE, zorder=5,
    #             label=f"Clearing point ({total_demand:.1f} MW, ${mcp:.2f})")

    #     # --- Formatting ---
    #     ax.set_xlabel("Cumulative Capacity (MW)", fontsize=12)
    #     ax.set_ylabel("Bid Price ($/MWh)", fontsize=12)
    #     ax.set_title("Day-Ahead Market — Merit Order Curve (Model 6)", fontsize=14, fontweight="bold")
    #     ax.set_xlim(0, max(total_capacity, x_cursor_d) * 1.02)
    #     ax.set_ylim(-0.5, max(all_costs) * 2 + 1)
    #     ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7, zorder=1)

    #     # --- Legend ---
    #     legend_patches = [
    #         mpatches.Patch(facecolor=COLOR_WIND,        hatch="///", edgecolor="black", label="Wind (zero marginal cost)"),
    #         mpatches.Patch(facecolor=COLOR_FORCED,      hatch="xxx", edgecolor="black", label="Forced reserve minimum (bid $0)"),
    #         mpatches.Patch(facecolor=COLOR_FREE,        hatch="...", edgecolor="black", label="Zero-cost conventional"),
    #         mpatches.Patch(facecolor=COLOR_DISPATCHED,  edgecolor="white", label="Dispatched (flexible capacity)"),
    #         mpatches.Patch(facecolor=COLOR_UNDISPATCHED,edgecolor="white", label="Not dispatched"),
    #     ]
    #     handles, _ = ax.get_legend_handles_labels()
    #     ax.legend(handles=legend_patches + handles, fontsize=9,
    #             loc="upper right", framealpha=0.9)

    #     plt.tight_layout()
    #     plt.savefig("code/results/step6_merit_order.png", dpi=150)
    #     return fig, ax

    # #def _plot_merit_order(self):
    #     """
    #     Plots the merit order curve for the Day-Ahead Market.
    #     Bars represent generator capacities stacked by bid price (ascending),
    #     with wind at zero cost. The market clearing price and total demand
    #     are shown as horizontal and vertical lines respectively.
    #     """
    #     mcp = self.out_dict["day_ahead_market"]["market_clearing_price_DAM"]

    #     # --- Build generator stack (conventional) ---
    #     # Use DAM-effective capacity (P_max minus reserved up-regulation)
    #     reserve_up = {i: self.combined_model.getVarByName(f"r_gen_up_{i}").X for i in self.P_gens}
    #     gen_names = list(self.P_gens.keys())
    #     gen_costs = [self.P_gens[i]["C"] for i in gen_names]
    #     gen_caps  = [self.P_gens[i]["P_max"] - reserve_up[i] for i in gen_names]

    #     # Sort conventional generators by bid price (merit order)
    #     sorted_gens = sorted(zip(gen_costs, gen_caps, gen_names), key=lambda x: x[0])
    #     sorted_costs, sorted_caps, sorted_names = zip(*sorted_gens)

    #     # --- Wind generators (zero marginal cost, prepend to merit order) ---
    #     wind_names = list(self.P_winds.keys())
    #     wind_caps  = [self.P_winds[i]["P_max"] * self.CF_wind[self.hour] for i in wind_names]
    #     total_wind = sum(wind_caps)

    #     # --- Total demand ---
    #     total_demand = self.Load_sum[self.hour]

    #     # --- Build bar positions ---
    #     # Wind block first (zero cost), then conventional in merit order
    #     all_caps   = [total_wind] + list(sorted_caps)
    #     all_costs  = [0.0]        + list(sorted_costs)
    #     all_labels = ["Wind"]     + list(sorted_names)

    #     # Determine which units are dispatched (scheduled output > 0)
    #     dispatched_set = set()
    #     scheduled = self.out_dict["day_ahead_market"]["scheduled_power_output"]
    #     gen_keys  = list(self.P_gens.keys())
    #     wind_keys = list(self.P_winds.keys())
    #     for key in gen_keys:
    #         if scheduled.get(f"gen_{key}", 0.0) > 1e-6:
    #             dispatched_set.add(key)
    #     if any(scheduled.get(f"wind_{key}", 0.0) > 1e-6 for key in wind_keys):
    #         dispatched_set.add("Wind")

    #     # --- Plot ---
    #     fig, ax = plt.subplots(figsize=(13, 6))

    #     x_cursor = 0.0
    #     bar_centers = []
    #     bar_colors  = []

    #     COLOR_WIND        = "#2196F3"
    #     COLOR_DISPATCHED  = "#4CAF50"
    #     COLOR_UNDISPATCHED= "#BDBDBD"
    #     COLOR_MCP_LINE    = "#E53935"
    #     COLOR_DEMAND_LINE = "#FF6F00"

    #     for cap, cost, label in zip(all_caps, all_costs, all_labels):
    #         if cap < 1e-6:
    #             continue
    #         is_wind = (label == "Wind")
    #         key_for_dispatch = label  # "Wind" or gen key
    #         dispatched = (key_for_dispatch in dispatched_set) or is_wind

    #         color = COLOR_WIND if is_wind else (COLOR_DISPATCHED if dispatched else COLOR_UNDISPATCHED)
    #         bar_centers.append(x_cursor + cap / 2)
    #         bar_colors.append(color)

    #         ax.bar(
    #             x_cursor + cap / 2, cost,
    #             width=cap,
    #             color=color,
    #             edgecolor="white",
    #             linewidth=0.6,
    #             align="center",
    #             zorder=2
    #         )
    #         # Label bar with generator name (rotate if narrow)
    #         label_str = str(label)
    #         if cap > 15:
    #             ax.text(
    #                 x_cursor + cap / 2, cost + 0.15,
    #                 f"G{label}" if label != "Wind" else "Wind 1-6",
    #                 ha="center", va="bottom", fontsize=7.5, rotation=0, color="black"
    #             )
    #         x_cursor += cap

    #     total_capacity = x_cursor

    #     # --- Step line for bid prices (merit order curve) ---
    #     step_x = []
    #     step_y = []
    #     x_cursor2 = 0.0
    #     for cap, cost, label in zip(all_caps, all_costs, all_labels):
    #         if cap < 1e-6:
    #             continue
    #         step_x.append(x_cursor2)
    #         step_y.append(cost)
    #         x_cursor2 += cap
    #         step_x.append(x_cursor2)
    #         step_y.append(cost)

    #     ax.step(
    #         step_x, step_y,
    #         where="pre",
    #         color="black", linewidth=1.4, linestyle="-",
    #         label="Bid price (merit order)", zorder=3
    #     )

    #     # --- Market clearing price (horizontal line) ---
    #     ax.axhline(
    #         mcp,
    #         color=COLOR_MCP_LINE, linewidth=1.8, linestyle="--",
    #         label=f"Market clearing price = \${mcp:.2f}/MWh",
    #         zorder=4
    #     )

    #     # --- Total demand (vertical line) ---
    #     ax.axvline(
    #         total_demand,
    #         color=COLOR_DEMAND_LINE, linewidth=1.8, linestyle="--",
    #         label=f"Total demand = {total_demand:.1f} MW",
    #         zorder=4
    #     )

    #     # --- Demand merit order curve ---
    #     demand_keys = list(self.Demands["load_distribution"].keys())
    #     demand_quantities = [
    #         self.Demands["load_distribution"][i] * self.Load_sum[self.hour]
    #         for i in demand_keys
    #     ]
    #     demand_prices = [self.Demands["bidding_prices"][i] for i in demand_keys]

    #     # Sort descending by bid price (highest willingness to pay first)
    #     sorted_demands = sorted(
    #         zip(demand_prices, demand_quantities, demand_keys),
    #         key=lambda x: x[0],
    #         reverse=True
    #     )
    #     sorted_d_prices, sorted_d_quantities, sorted_d_keys = zip(*sorted_demands)

    #     # Build staircase
    #     demand_step_x = []
    #     demand_step_y = []
    #     x_cursor_d = 0.0
    #     for qty, price in zip(sorted_d_quantities, sorted_d_prices):
    #         if qty < 1e-6:
    #             continue
    #         demand_step_x.append(x_cursor_d)
    #         demand_step_y.append(price)
    #         x_cursor_d += qty
    #         demand_step_x.append(x_cursor_d)
    #         demand_step_y.append(price)

    #     ax.step(
    #         demand_step_x, demand_step_y,
    #         where="pre",
    #         color=COLOR_DEMAND_LINE, linewidth=1.8, linestyle="-",
    #         label="Demand bid curve", zorder=3
    #     )

    #     # --- Intersection marker ---
    #     ax.plot(
    #         total_demand, mcp,
    #         marker="o", markersize=9,
    #         color=COLOR_MCP_LINE, zorder=5,
    #         label=f"Clearing point ({total_demand:.1f} MW, \${mcp:.2f})"
    #     )

    #     # --- Axes labels & formatting ---
    #     ax.set_xlabel("Cumulative Capacity (MW)", fontsize=12)
    #     ax.set_ylabel("Bid Price ($/MWh)", fontsize=12)
    #     ax.set_title("Day-Ahead Market — Merit Order Curve", fontsize=14, fontweight="bold")
    #     ax.set_xlim(0, max(total_capacity, x_cursor_d) * 1.02)
    #     ax.set_ylim(-0.5, max(all_costs) * 2 + 1)
    #     ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7, zorder=1)

    #     # --- Legend ---
    #     legend_patches = [
    #         mpatches.Patch(color=COLOR_WIND,         label="Wind (zero marginal cost)"),
    #         mpatches.Patch(color=COLOR_DISPATCHED,   label="Dispatched (conventional)"),
    #         mpatches.Patch(color=COLOR_UNDISPATCHED, label="Not dispatched"),
    #     ]
    #     handles, labels_leg = ax.get_legend_handles_labels()
    #     ax.legend(
    #         handles=legend_patches + handles,
    #         fontsize=9, loc="upper right", framealpha=0.9
    #     )

    #     plt.tight_layout()
    #     plt.savefig("code/results/step6_merit_order.png", dpi=150)
    #     #plt.show()
    #     return fig, ax


