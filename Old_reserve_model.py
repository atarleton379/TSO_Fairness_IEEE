import gurobipy as gp
from gurobipy import GRB

from model_helpers import load_input_file


class Standard_Reserve_Model:
    """
    Original Reserve sharing model from first iteration of white paper ##(might include some logical errors regarding negated flow etc.)
    Includes: 
    -non-neighbour sharing
    -ad-hoc fairness
    -tiebreaker

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
        m = gp.Model("standard_reserve")
        if not verbose:
            m.Params.OutputFlag = 0

        zones = sorted(self.Zones.keys())

        r = {}
        f = {}
        sa = {}
        sb = {}

        for z in zones:
            r[z] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=self.RI_by_zone[z], name=f"reserve_zone_{z}")
            sa[z] = m.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0,
                ub=sum(self.Atc[(v, z)] for v in zones if v != z),
                name=f"sharing_available_{z}",
            )
            sb[z] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=self.RI_by_zone[z], name=f"sharing_benefit_{z}")

        for (u, v), cap in self.Atc.items():
            if u != v:
                f[(u, v)] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=cap, name=f"sharing_flow_{u}_{v}")

        m.setObjective(gp.quicksum(r[z] for z in zones), GRB.MINIMIZE)

        m.addConstrs((r[z] + sa[z] >= self.RI_by_zone[z] for z in zones), name="reserve_plus_sharing_covers_reference_incident")
        m.addConstrs((sb[z] == self.RI_by_zone[z] - r[z] for z in zones), name="define_sharing_benefit")
        m.addConstrs((sa[z] == gp.quicksum(f[(v, z)] for v in zones if v != z) for z in zones), name="define_sharing_available")
        m.addConstrs((gp.quicksum(f[(z, v)] for v in zones if v != z) <= r[z] for z in zones), name="limit_sharing_to_own_procurement")

        m.optimize()

        self.model = m
        if m.Status == GRB.OPTIMAL:
            self.out_dict = {
                "status": int(m.Status),
                "objective": float(m.ObjVal),
                "r": {z: float(r[z].X) for z in zones},
                "sa": {z: float(sa[z].X) for z in zones},
                "sb": {z: float(sb[z].X) for z in zones},
                "flow": {f"{u}->{v}": float(var.X) for (u, v), var in f.items() if abs(var.X) > 1e-9},
            }
        else:
            self.out_dict = {"status": int(m.Status), "objective": None, "r": {}, "sa": {}, "sb": {}, "flow": {}}

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
