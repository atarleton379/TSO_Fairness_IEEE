import gurobipy as gp
from gurobipy import GRB

from model_helpers import load_input_file


class MMF_Standard_Reserve_Model:
    """
    MaxMin Fairness Reserve Sharing model based on second iteration of white paper
    Includes: 
    -only neighbour sharing
    -maximizing responsibility
    -no tiebreaker

    You provide:
    - inp_hndl (object with at least .zones)
    - atc (dict keyed by (u, v))
    - RI_by_zone (dict keyed by zone)

    Optional convenience:
    - Use from_input_file(...) to load inp_hndl from a JSON file.
    """

    def __init__(self, hour: int, inp_hndl, atc: dict, RI_by_zone: dict):
        self.hour = hour
        self.Zones = inp_hndl.zones
        self.Atc = atc
        self.RI_by_zone = RI_by_zone
        self.model = None
        self.out_dict = {}

    @classmethod
    def from_input_file(cls, hour: int, input_file: str, atc: dict, RI_by_zone: dict):
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
        return cls(hour=hour, inp_hndl=inp_hndl, atc=atc, RI_by_zone=RI_by_zone)

    def solve(self, verbose: bool = False):
        ## Define Model
        m = gp.Model("model_adjacent")

        ## Define variables
        r = {}
        f = {}
        sa = {}
        sb = {}

        for z in self.Zones:
            ## Define required reserve procurement in each zone
            r[z] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=self.RI_by_zone[z], name=f"reserve_zone_{z}")

            ## Define sharing available to each zone
            sa[z] = m.addVar( 
                vtype=GRB.CONTINUOUS,
                lb=0,
                ub=sum(self.Atc[(v, z)] for v in self.Zones if v != z),
                name=f"sharing_available_{z}"
            )

            ## Define sharing benefit to each zone
            sb[z] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"sharing_benefit_{z}")

        # Directed sharing flow: flow[u, v] means export from zone u to zone v
        for (u, v), cap in self.Atc.items():
            if u != v:
                f[u, v] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=cap, name=f"sharing_flow_{u}_{v}")

        ## Define Objective
        objective = gp.quicksum(self.RI_by_zone[z] - r[z] for z in self.Zones)
        m.setObjective(objective, GRB.MAXIMIZE)

        ## Define Constraints

        # Reserve plus sharing must cover the reference incident
        m.addConstrs((r[z] + sa[z] >= self.RI_by_zone[z] for z in self.Zones), name="reserve_plus_sharing_covers_reference_incident")

        # Benefit equals avoided own procurement
        m.addConstrs((sb[z] == self.RI_by_zone[z] - r[z] for z in self.Zones), name="define_sharing_benefit")

        # Sharing available is the incoming flow into the zone
        m.addConstrs((sa[z] == gp.quicksum(f[v, z] for v in self.Zones if v != z) for z in self.Zones), name="define_sharing_available")

        # A zone cannot export more than it procures
        m.addConstrs((f[z, v] <= r[z] for z in self.Zones for v in self.Zones if v != z ), name="limit_sharing_to_own_procurement")
        ## Lexicographic max-min fairness
        
        remaining = set(self.Zones.keys())
        tier_values = {}
        tier_index = 1
        tol = 1e-6

        while remaining:
            # Maximize the minimum utility among the remaining zones.
            t = m.addVar(lb=-GRB.INFINITY, name=f"tier_min_utility_{tier_index}")
            m.addConstrs((t <= sb[z] for z in remaining), name=f"tier_floor_{tier_index}")
            m.setObjective(t + 0.0001 * gp.quicksum(sb[z] for z in remaining), GRB.MAXIMIZE)

            m.optimize()
            if m.Status != GRB.OPTIMAL:
                break

            t_star = t.X

            # Zones at the current bottleneck utility are fixed for subsequent tiers.
            newly_fixed = [z for z in remaining if sb[z].X <= t_star + tol]
            if not newly_fixed:
                # Numerical fallback: always fix at least one zone.
                z_min = min(remaining, key=lambda z: sb[z].X)
                newly_fixed = [z_min]

            for z in newly_fixed:
                m.addConstr(sb[z] == t_star, name=f"fix_tier_{tier_index}_zone_{z}")
                tier_values[z] = t_star
                remaining.remove(z)

            tier_index += 1

        self.model = m
        if m.Status == GRB.OPTIMAL:
            self.out_dict = {
                "objective": m.ObjVal,
                "tier_values": tier_values,
                "reserve": {z: r[z].X for z in self.Zones},
                "sharing_available": {z: sa[z].X for z in self.Zones},
                "sharing_benefit": {z: sb[z].X for z in self.Zones},
                "sharing_flows": {(u, v): f[u, v].X for (u, v) in f},
                "total_savings": sum(sb[z].X for z in self.Zones),
                "total_procurement": sum(r[z].X for z in self.Zones)
            }
            print("Lexicographic max-min fairness solved.")
            print("Tier-fixed utilities:", tier_values)
            print("reserve:", self.out_dict["reserve"])
            print("sharing_benefit:", self.out_dict["sharing_benefit"])
            print("sharing_flows:", self.out_dict["sharing_flows"])
        else:
            self.out_dict = {"status": int(m.Status)}
            print(f"Solver status: {m.Status}")
        return self.out_dict


def run_MMF_standard_reserve_from_file(
    input_file: str,
    hour: int,
    atc: dict,
    RI_by_zone: dict,
    verbose: bool = False,
):
    """
    Convenience wrapper for plug-and-play runs.

    Example:
    out = run_standard_reserve_from_file(
        input_file="inputs/case_a.json",
        hour=12,
        atc=atc,
        RI_by_zone=RI_by_zone,
    )
    """
    inp_hndl = load_input_file(input_file)
    model = MMF_Standard_Reserve_Model(hour=hour, inp_hndl=inp_hndl, atc=atc, RI_by_zone=RI_by_zone)
    return model.solve(verbose=verbose)

def main():
    from input import InputHandler
    from model_helpers import ATC_calc, reference_incident_by_zone

    hour = 12
    inp_hndl = InputHandler()
    atc = ATC_calc(inp_hndl.zones, inp_hndl.lines, symmetric=False)
    ri_by_zone = reference_incident_by_zone(inp_hndl)

    model = MMF_Standard_Reserve_Model(hour=hour, inp_hndl=inp_hndl, atc=atc, RI_by_zone=ri_by_zone)
    results = model.solve()

    

if __name__ == "__main__":
    main()
    
