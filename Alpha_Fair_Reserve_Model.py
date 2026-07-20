import gurobipy as gp
import pandas as pd
from gurobipy import GRB

from model_helpers import load_input_file


class Alpha_Fair_Reserve_Model:
    """
    Alpha-Fairness Reserve Sharing model based on second iteration of white paper.

    Generalizes Standard_Reserve_Model / PF_Standard_Reserve_Model /
    MMF_Standard_Reserve_Model via the alpha-fair utility function applied to
    each zone's sharing benefit (avoided own procurement):

        U_alpha(x) = x^(1 - alpha) / (1 - alpha)   for alpha != 1
        U_alpha(x) = log(x)                         for alpha == 1

    Special cases (for intuition, not implemented via those other classes):
    - alpha = 0        -> utilitarian sum (equivalent in spirit to Standard_Reserve_Model)
    - alpha = 1        -> proportional fairness (equivalent to PF_Standard_Reserve_Model)
    - alpha -> infinity -> max-min fairness (approaches MMF_Standard_Reserve_Model)

    Includes:
    -only neighbour sharing
    -maximizing alpha-fair utility of each zone's sharing benefit
    -no tiebreaker

    You provide:
    - inp_hndl (object with at least .zones)
    - atc (dict keyed by (u, v))
    - RI_by_zone (dict keyed by zone)
    - alpha (float, fairness parameter; default 2.0)

    Optional convenience:
    - Use from_input_file(...) to load inp_hndl from a JSON file.

    Note on duals: for alpha == 0 the model is a pure LP and duals/reduced
    costs are extracted the same way as Standard_Reserve_Model. For any other
    alpha, the utility is implemented via Gurobi general constraints
    (addGenConstrLog / addGenConstrPow), which are solved as a MIP internally
    (piecewise-linear approximation). Gurobi does not provide LP duals for
    MIPs, so duals_table will be None in that case.
    """

    def __init__(self, hour: int, inp_hndl, atc: dict, RI_by_zone: dict, alpha: float = 2.0):
        self.hour = hour
        self.Zones = inp_hndl.zones
        self.Atc = atc
        self.RI_by_zone = RI_by_zone
        self.alpha = alpha
        self.model = None
        self.out_dict = {}
        self.duals_table = None

    @classmethod
    def from_input_file(cls, hour: int, input_file: str, atc: dict, RI_by_zone: dict, alpha: float = 2.0):
        """
        Load a simple scenario JSON and build the model object.

        Required JSON key:
        - zones

        Other keys are allowed and ignored by this model.
        """
        inp_hndl = load_input_file(input_file)
        if not hasattr(inp_hndl, "zones"):
            raise ValueError("Input file must contain 'zones'.")
        inp_hndl.zones = {int(k): v for k, v in inp_hndl.zones.items()}
        return cls(hour=hour, inp_hndl=inp_hndl, atc=atc, RI_by_zone=RI_by_zone, alpha=alpha)

    def solve(self, verbose: bool = False):
        ## Define Model
        m = gp.Model("model_alpha_fair")

        ## Define variables
        r = {}
        f = {}
        sa = {}
        sb = {}

        for z in self.Zones:
            ## Define required reserve procurement in each zone
            r[z] = m.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0,
                ub=self.RI_by_zone[z],
                name=f"reserve_zone_{z}",
            )

            ## Define sharing available to each zone
            sa[z] = m.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0,
                ub=sum(self.Atc.get((v, z), 0.0) for v in self.Zones if v != z),
                name=f"sharing_available_{z}"
            )

            ## Define sharing benefit to each zone
            sb[z] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"sharing_benefit_{z}")

        # Directed sharing flow: flow[u, v] means export from zone u to zone v
        for (u, v), cap in self.Atc.items():
            if u != v:
                f[u, v] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=cap, name=f"sharing_flow_{u}_{v}")

        ## Define Objective: maximize sum_z U_alpha(sharing benefit of zone z)
        alpha = self.alpha
        eps = 1e-6
        is_pure_lp = abs(alpha) < 1e-9

        if is_pure_lp:
            # alpha = 0: utility is the identity, so this stays a plain LP.
            objective = gp.quicksum(self.RI_by_zone[z] - r[z] for z in self.Zones)
        elif abs(alpha - 1.0) < 1e-9:
            # alpha = 1: proportional fairness via log utility.
            obj_term_pos = {}
            obj_term_util = {}
            for z in self.Zones:
                obj_term_pos[z] = m.addVar(lb=eps, name=f"obj_term_pos_{z}")
                obj_term_util[z] = m.addVar(lb=-GRB.INFINITY, name=f"obj_term_util_{z}")
                m.addConstr(obj_term_pos[z] == self.RI_by_zone[z] - r[z] + eps, name=f"obj_term_shift_{z}")
                m.addGenConstrLog(obj_term_pos[z], obj_term_util[z], name=f"obj_term_log_link_{z}")
            objective = gp.quicksum(obj_term_util[z] for z in self.Zones)
        else:
            # General alpha: U_alpha(x) = x^(1 - alpha) / (1 - alpha).
            p = 1.0 - alpha
            obj_term_pos = {}
            obj_term_pow = {}
            for z in self.Zones:
                obj_term_pos[z] = m.addVar(lb=eps, name=f"obj_term_pos_{z}")
                obj_term_pow[z] = m.addVar(lb=-GRB.INFINITY, name=f"obj_term_pow_{z}")
                m.addConstr(obj_term_pos[z] == self.RI_by_zone[z] - r[z] + eps, name=f"obj_term_shift_{z}")
                m.addGenConstrPow(obj_term_pos[z], obj_term_pow[z], p, name=f"obj_term_pow_link_{z}")
            objective = gp.quicksum(obj_term_pow[z] / p for z in self.Zones)

        if not is_pure_lp:
            # Tighter-than-default piecewise-linear approximation of the
            # nonlinear utility functions (log / power) used above.
            m.Params.FuncPieces = -1
            m.Params.FuncPieceError = 1e-4

        m.setObjective(objective, GRB.MAXIMIZE)

        ## Define Constraints

        # Reserve plus sharing must cover the reference incident
        c_reference_incident = m.addConstrs((r[z] + sa[z] >= self.RI_by_zone[z] for z in self.Zones), name="reserve_plus_sharing_covers_reference_incident")

        # Benefit equals avoided own procurement
        c_sharing_benefit = m.addConstrs((sb[z] == self.RI_by_zone[z] - r[z] for z in self.Zones), name="define_sharing_benefit")

        # Sharing available is the incoming flow into the zone
        c_sharing_available = m.addConstrs((sa[z] == gp.quicksum(f[v, z] for v in self.Zones if v != z) for z in self.Zones), name="define_sharing_available")

        # A zone cannot export more than it procures
        c_limit_sharing = m.addConstrs((f[z, v] <= r[z] for z in self.Zones for v in self.Zones if v != z ), name="limit_sharing_to_own_procurement")

        ## Solve model
        m.optimize()

        ## Read results
        if m.Status == GRB.OPTIMAL:
            self.model = m

            duals_df = None
            if is_pure_lp:
                try:
                    duals_df = self._build_duals_table(
                        r=r, sa=sa, f=f,
                        c_reference_incident=c_reference_incident,
                        c_sharing_benefit=c_sharing_benefit,
                        c_sharing_available=c_sharing_available,
                        c_limit_sharing=c_limit_sharing,
                    )
                except (AttributeError, gp.GurobiError):
                    duals_df = None
            self.duals_table = duals_df

            self.out_dict = {
                "objective": m.ObjVal,
                "alpha": alpha,
                "reserve": {z: r[z].X for z in self.Zones},
                "sharing_available": {z: sa[z].X for z in self.Zones},
                "sharing_benefit": {z: sb[z].X for z in self.Zones},
                "sharing_flows": {(u, v): f[u, v].X for (u, v) in f},
                "total_savings": sum(sb[z].X for z in self.Zones),
                "total_procurement": sum(r[z].X for z in self.Zones)
            }
            if duals_df is not None:
                self.out_dict["duals_by_zone"] = duals_df.to_dict(orient="records")

            print(f"Optimal objective (alpha={alpha}): {m.ObjVal}")
            print("r:", {z: r[z].X for z in self.Zones})
            print("sa:", {z: sa[z].X for z in self.Zones})
            print("sb:", {z: sb[z].X for z in self.Zones})
            if verbose:
                if duals_df is not None:
                    print("\nDual values / reduced costs by zone:")
                    print(duals_df.to_string(index=False))
                else:
                    print(
                        "\nDuals not available: the alpha-fair objective uses nonlinear "
                        "(piecewise-linear) general constraints for this alpha, so Gurobi "
                        "solves it as a MIP and does not provide LP duals/reduced costs."
                    )
        else:
            self.model = m
            self.out_dict = {"status": int(m.Status)}
            print(f"Solver status: {m.Status}")
        return self.out_dict

    def _build_duals_table(self, r, sa, f, c_reference_incident, c_sharing_benefit,
                            c_sharing_available, c_limit_sharing, tol: float = 1e-6):
        """
        Build a long-format table of duals (for constraints) and reduced costs
        (for variable bounds), indexed by zone. One row per constraint/bound
        instance so it can be filtered or grouped by "zone" in pandas.

        Only meaningful when the model solved as a pure LP (alpha == 0); see
        the class docstring for details.

        Columns:
        - zone: the zone the row is associated with
        - related_zone: neighboring zone involved (only for pairwise constraints)
        - constraint: name of the constraint or bound
        - type: relational type (">=", "<=", "==", "var_ub")
        - value: current value of the constrained expression / variable
        - bound: right-hand side / bound value
        - slack: distance from the bound (0 when binding)
        - dual: Pi (constraints) or RC (variable bounds)
        - binding: True if the constraint/bound is active at the solution
        """
        rows = []

        for z in self.Zones:
            # Reference incident coverage: r[z] + sa[z] >= RI[z]
            con = c_reference_incident[z]
            rows.append({
                "zone": z,
                "related_zone": None,
                "constraint": "reserve_plus_sharing_covers_reference_incident",
                "type": ">=",
                "value": r[z].X + sa[z].X,
                "bound": self.RI_by_zone[z],
                "slack": con.Slack,
                "dual": con.Pi,
                "binding": abs(con.Slack) < tol,
            })

            # Sharing benefit definition: sb[z] == RI[z] - r[z]
            con = c_sharing_benefit[z]
            rows.append({
                "zone": z,
                "related_zone": None,
                "constraint": "define_sharing_benefit",
                "type": "==",
                "value": self.RI_by_zone[z] - r[z].X,
                "bound": self.RI_by_zone[z] - r[z].X,
                "slack": con.Slack,
                "dual": con.Pi,
                "binding": True,
            })

            # Sharing available definition: sa[z] == sum of inbound flows
            con = c_sharing_available[z]
            rows.append({
                "zone": z,
                "related_zone": None,
                "constraint": "define_sharing_available",
                "type": "==",
                "value": sa[z].X,
                "bound": sa[z].X,
                "slack": con.Slack,
                "dual": con.Pi,
                "binding": True,
            })

            # Variable upper bound: r[z] <= RI[z]
            rows.append({
                "zone": z,
                "related_zone": None,
                "constraint": "reserve_upper_bound",
                "type": "var_ub",
                "value": r[z].X,
                "bound": r[z].UB,
                "slack": r[z].UB - r[z].X,
                "dual": r[z].RC,
                "binding": abs(r[z].UB - r[z].X) < tol,
            })

            # Variable upper bound: sa[z] <= sum of inbound ATC
            rows.append({
                "zone": z,
                "related_zone": None,
                "constraint": "sharing_available_upper_bound",
                "type": "var_ub",
                "value": sa[z].X,
                "bound": sa[z].UB,
                "slack": sa[z].UB - sa[z].X,
                "dual": sa[z].RC,
                "binding": abs(sa[z].UB - sa[z].X) < tol,
            })

            # Pairwise constraints/bounds: one set per neighboring zone v
            for v in self.Zones:
                if v == z:
                    continue

                # Skip pairs with no flow in either direction; their duals
                # are still computed by the solver but are not useful to report.
                flow_zv = f[z, v].X if (z, v) in f else 0.0
                flow_vz = f[v, z].X if (v, z) in f else 0.0
                if abs(flow_zv) < tol and abs(flow_vz) < tol:
                    continue

                # Export limit: f[z, v] <= r[z]
                if (z, v) in c_limit_sharing:
                    con = c_limit_sharing[z, v]
                    rows.append({
                        "zone": z,
                        "related_zone": v,
                        "constraint": "limit_sharing_to_own_procurement",
                        "type": "<=",
                        "value": f[z, v].X,
                        "bound": r[z].X,
                        "slack": con.Slack,
                        "dual": con.Pi,
                        "binding": abs(con.Slack) < tol,
                    })

                # Flow capacity bound: f[z, v] <= Atc[z, v]
                if (z, v) in f:
                    rows.append({
                        "zone": z,
                        "related_zone": v,
                        "constraint": "sharing_flow_upper_bound",
                        "type": "var_ub",
                        "value": f[z, v].X,
                        "bound": f[z, v].UB,
                        "slack": f[z, v].UB - f[z, v].X,
                        "dual": f[z, v].RC,
                        "binding": abs(f[z, v].UB - f[z, v].X) < tol,
                    })

        df = pd.DataFrame(rows)
        return df.sort_values(["zone", "constraint", "related_zone"], na_position="first").reset_index(drop=True)

    def _wide_duals_table(self, value_col: str = "dual") -> pd.DataFrame:
        """
        Reshape the long-format duals table into a zone-as-column CSV layout.

        - Each zone gets its own column (named by its zone id).
        - Constraints that exist once per zone (no related_zone) produce a
          single row, with each zone's value under its own column.
        - Constraints that exist per directional edge (related_zone is set,
          e.g. sharing flow limits) produce one row per zone (the "row zone"),
          where the cell under column z holds the value for the flow directed
          from zone z (column) to the row zone (the flow's destination).

        Columns:
        - constraint: constraint/bound name
        - row_zone: the destination zone for per-edge rows, else blank
        - <zone id> (one column per zone): value for that zone/edge
        """
        if self.duals_table is None or self.duals_table.empty:
            raise ValueError(
                "No duals table available. Either run solve() first, or this "
                "alpha value produces a nonlinear (MIP) formulation for which "
                "Gurobi does not provide LP duals (see class docstring)."
            )

        duals_df = self.duals_table
        zone_cols = sorted(int(z) for z in duals_df["zone"].dropna().unique())

        # Preserve the order constraints were first encountered.
        constraint_order = list(dict.fromkeys(duals_df["constraint"]))

        rows = []
        for constraint in constraint_order:
            sub = duals_df[duals_df["constraint"] == constraint]
            is_per_edge = sub["related_zone"].notna().any()

            if not is_per_edge:
                row: dict = {"constraint": constraint, "row_zone": None}
                for z in zone_cols:
                    match = sub[sub["zone"] == z]
                    row[z] = match[value_col].iloc[0] if not match.empty else None
                rows.append(row)
            else:
                # One row per destination ("row") zone; columns are the
                # origin ("column") zone the flow comes from.
                for row_zone in zone_cols:
                    row: dict = {"constraint": constraint, "row_zone": row_zone}
                    for col_zone in zone_cols:
                        if col_zone == row_zone:
                            row[col_zone] = None
                            continue
                        match = sub[(sub["zone"] == col_zone) & (sub["related_zone"] == row_zone)]
                        row[col_zone] = match[value_col].iloc[0] if not match.empty else None
                    rows.append(row)

        wide_df = pd.DataFrame(rows)
        return wide_df[["constraint", "row_zone"] + zone_cols]

    def save_duals_csv(self, path: str, value_col: str = "dual") -> pd.DataFrame:
        """
        Write the duals table to a CSV in zones-as-columns format and return
        the resulting DataFrame.

        value_col selects which field populates the cells: "dual" (default,
        Pi for constraints / RC for variable bounds), "value", "slack",
        or "binding".
        """
        wide_df = self._wide_duals_table(value_col=value_col)
        wide_df.to_csv(path, index=False)
        return wide_df


def run_alpha_fair_reserve_from_file(
    input_file: str,
    hour: int,
    atc: dict,
    RI_by_zone: dict,
    alpha: float = 2.0,
    verbose: bool = False,
):
    """
    Convenience wrapper for plug-and-play runs.

    Example:
    out = run_alpha_fair_reserve_from_file(
        input_file="inputs/case_a.json",
        hour=12,
        atc=atc,
        RI_by_zone=RI_by_zone,
        alpha=2.0,
    )
    """
    inp_hndl = load_input_file(input_file)
    model = Alpha_Fair_Reserve_Model(hour=hour, inp_hndl=inp_hndl, atc=atc, RI_by_zone=RI_by_zone, alpha=alpha)
    return model.solve(verbose=verbose)
