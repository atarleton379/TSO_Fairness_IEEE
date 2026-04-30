import gurobipy as gp
from gurobipy import GRB
import time

from src.input import InputHandler


class Model3Zonal:
    """
    Single-hour day-ahead market clearing incl. network constraints. Nodes grouped into zones.
    """

    def __init__(self, hour: int, inp_hndl: InputHandler, sensitivity_factor: float):
        """
        Initializes and runs the network-constrained optimal power flow model
        for a given hour. Zonal mdodel.

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
            Zones (dict): Dictionary detailing which nodes belong to which zone.
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
        self.Zones = inp_hndl.zones

        self.model = self._run_model()
        if self.model.status == gp.GRB.OPTIMAL:
            self.out_dict = self._analyze_output()

    def _run_model(self) -> gp.Model:
        """
        Builds and solves a zonal market clearing model with inter-zonal
        transmission constraints for the current hour.

        Aggregates the nodal network into zones, computing inter-zonal transfer
        capacities by summing the scaled capacities of all cross-zonal lines per
        zone pair. Intra-zonal lines are ignored. The model clears the market
        subject to zonal power balance and inter-zonal flow bounds.

        Pre-processing:
            - Builds a node-to-zone lookup (stored as self.node_zone).
            - Aggregates inter-zonal line capacities by zone pair, scaled by
              sensitivity_factor (stored as self.zone_pairs).
            - Zone pairs are keyed as (min_zone, max_zone) to avoid duplicates.

        Decision Variables:
            p_gen[i]               (continuous, >= 0):       Output of conventional generator i.
            p_wind[i]              (continuous, >= 0):       Output of wind farm i.
            demand[i]              (continuous, >= 0):       Served demand for load i.
            p_zone_connections[pair] (continuous, [-cap, cap]): Net power flow between zone pair,
                                                                where cap is the aggregated
                                                                inter-zonal line capacity.

        Objective:
            Maximize social welfare:
                sum_i(demand[i] * bidding_price[i]) - sum_i(C[i] * p_gen[i])

        Constraints:
            - Max generation:    p_gen[i]  <= P_max[i]                           for all i in P_gens
            - Max wind output:   p_wind[i] <= P_max[i] * CF_wind[hour]           for all i in P_winds
            - Max demand:        demand[i] <= load_distribution[i] * Load[hour]  for all i in Demands
            - Zonal balance:     sum(inflows) - sum(outflows)
                                 + sum(generation) + sum(wind) - sum(demand) == 0  for all n in Zones

        Returns:
            gp.Model: The Gurobi model after optimization, regardless of solve status.
                Use model.status == GRB.OPTIMAL to check for a valid solution.
        """
        model = gp.Model("model1")

        p_gen = {}
        p_wind = {}
        demand = {}
        p_zone_connections = {}

        # Build node -> zone lookup
        node_zone = {}
        for z, nodes in self.Zones.items():
            for n in nodes:
                node_zone[n] = z
        self.node_zone = node_zone

        # Sum capacity of inter-zonal lines by zone pair (ignores intrazonal lines), scaled by sensitivity factor
        zone_connection_capacity = {}
        for _, line in self.Lines.items():
            from_zone = node_zone[line["from"]]
            to_zone = node_zone[line["to"]]
            if from_zone == to_zone:
                continue  # intrazonal line, skip
            pair = (min(from_zone, to_zone), max(from_zone, to_zone))
            zone_connection_capacity[pair] = (
                zone_connection_capacity.get(pair, 0)
                + line["capacity"] * self.sensitivity_factor
            )
        self.zone_pairs = list(zone_connection_capacity.keys())

        # Define Variables
        for i in self.P_gens:
            p_gen[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"p_gen_{i}")
        for i in self.P_winds:
            p_wind[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"p_wind_{i}")
        for i in self.Demands["load_distribution"]:
            demand[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"demand_{i}")
        # Define Zone connection variables
        for pair, cap in zone_connection_capacity.items():
            p_zone_connections[pair] = model.addVar(
                vtype=GRB.CONTINUOUS, lb=-cap, ub=cap, name=f"zone_{pair[0]}_{pair[1]}"
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
        # zonal balance constraints
        for n in self.Zones:
            model.addConstr(
                -gp.quicksum(
                    p_zone_connections[(n, z)]
                    for z in self.Zones
                    if z > n and (n, z) in p_zone_connections
                )  # outflow to higher-numbered zones
                + gp.quicksum(
                    p_zone_connections[(z, n)]
                    for z in self.Zones
                    if z < n and (z, n) in p_zone_connections
                )  # inflow from lower-numbered zones
                + gp.quicksum(
                    p_gen[g]
                    for g in self.P_gens
                    if node_zone[self.P_gens[g]["node"]] == n
                )  # generation in zone n
                + gp.quicksum(
                    p_wind[w]
                    for w in self.P_winds
                    if node_zone[self.P_winds[w]["node"]] == n
                )  # wind generation in zone n
                - gp.quicksum(
                    demand[d]
                    for d in self.Demands["load_location"]
                    if node_zone[self.Demands["load_location"][d]] == n
                )  # load in zone n
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
        Extracts and computes zonal market results from the optimal zonal
        clearing solution.

        Retrieves primal variable values and dual prices (zonal prices) from
        the solved Gurobi model and computes economic metrics at the zone level,
        mapping each generator and wind farm to its zone via self.node_zone.

        Returns:
            dict: A dictionary containing the following market results:

                - zone_{n}_price (float):
                    Zonal marginal price for each zone n.

                - social_welfare (float):
                    Optimal objective value; total welfare across all zones.

                - zone_{a}_{b}_flow (float):
                    Realized net power flow for each inter-zonal connection (a, b),
                    where a < b. Positive values indicate flow from zone a to zone b.

                - total_power_consumed (float):
                    Sum of all cleared demands across all zones.

                - gen_profits (dict):
                    Individual profit for each conventional generator i:
                    p_gen[i] * (ZP[zone_i] - C[i]).

                - total_gen_profit (float):
                    Sum of profits across all conventional generators.

                - avg_gen_profit (float):
                    Average profit per conventional generator.

                - wind_profits (dict):
                    Individual profit for each wind farm i:
                    p_wind[i] * ZP[zone_i].

                - total_wind_profit (float):
                    Sum of profits across all wind farms.

                - avg_wind_profit (float):
                    Average profit per wind farm.
        """
        out_dict = {}
        out_dict["solve_start_perf_counter"] = getattr(self.model, "_solve_start_perf", None)
        out_dict["solve_end_perf_counter"] = getattr(self.model, "_solve_end_perf", None)
        out_dict["solve_time_seconds"] = getattr(self.model, "_solve_time_seconds", None)

        # zonal prices
        for n in self.Zones:
            out_dict[f"zone_{n}_price"] = self.model.getConstrByName(
                f"power_balance_{n}"
            ).Pi

        # social welfare
        out_dict["social_welfare"] = self.model.ObjVal

        # inter-zonal flows
        for pair in self.zone_pairs:
            out_dict[f"zone_{pair[0]}_{pair[1]}_flow"] = self.model.getVarByName(
                f"zone_{pair[0]}_{pair[1]}"
            ).X

        # demand per zone
        out_dict["total_power_consumed"] = sum(
            self.model.getVarByName(f"demand_{i}").X
            for i in self.Demands["load_location"]
        )

        # Conventional generator profits (individual, total, average)
        gen_profits = {
            i: self.model.getVarByName(f"p_gen_{i}").X
            * (
                -self.model.getConstrByName(
                    f"power_balance_{self.node_zone[self.P_gens[i]['node']]}"
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
                    f"power_balance_{self.node_zone[self.P_winds[i]['node']]}"
                ).Pi
            )
            for i in self.P_winds
        }
        out_dict["wind_profits"] = wind_profits
        out_dict["total_wind_profit"] = sum(wind_profits.values())
        out_dict["avg_wind_profit"] = out_dict["total_wind_profit"] / len(wind_profits)

        return out_dict
