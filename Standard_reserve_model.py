import gurobipy as gp
from gurobipy import GRB

from model_helpers import load_input_file


class Standard_Reserve_Model:
    """
    Reserve Sharing model based on second iteration of white paper
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

        ## Define Objective
        objective = gp.quicksum(self.RI_by_zone[z] - r[z] for z in self.Zones) + 0.0000001 * gp.quicksum(f.values())
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

        ## Solve model
        m.optimize()

        ## Read results
        if m.Status == GRB.OPTIMAL:
            self.model = m
            self.out_dict = {
                "objective": m.ObjVal,
                "reserve": {z: r[z].X for z in self.Zones},
                "sharing_available": {z: sa[z].X for z in self.Zones},
                "sharing_benefit": {z: sb[z].X for z in self.Zones},
                "sharing_flows": {(u, v): f[u, v].X for (u, v) in f},
            }
            print(f"Optimal objective: {m.ObjVal}")
            print("r:", {z: r[z].X for z in self.Zones})
            print("sa:", {z: sa[z].X for z in self.Zones})
            print("sb:", {z: sb[z].X for z in self.Zones})
        else:
            self.model = m
            self.out_dict = {"status": int(m.Status)}
            print(f"Solver status: {m.Status}")
        return self.out_dict


def run_standard_reserve_from_file(
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
    model = Standard_Reserve_Model(hour=hour, inp_hndl=inp_hndl, atc=atc, RI_by_zone=RI_by_zone)
    return model.solve(verbose=verbose)
