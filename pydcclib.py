"""P4 reserve-sharing: enumerate dual vertices with pycddlib, build the region catalog.

Pipeline:
  1. Build the dual polyhedron's H-representation (exact rational arithmetic).
  2. Enumerate vertices + rays with cdd.
  3. Project vertices to 10-dim price vectors eta = (coeffs on D_A..D_D, C_AB..C_DC).
  4. Dedupe identical price vectors (alphas vanish in projection -> duplicates).
  5. Sanity checks: two known anchor vertices, ray nonpositivity.
  6. Fuzz test: max_k eta_k . theta  ==  LP optimum, on random instances.

pycddlib API note: this uses the v3 API (cdd.gmp for exact arithmetic).
If you have v2.x, replace matrix_from_array/polyhedron_from_matrix with
cdd.Matrix(rows, number_type='fraction'), mat.rep_type = cdd.RepType.INEQUALITY,
cdd.Polyhedron(mat).get_generators().
"""

from fractions import Fraction
import cdd.gmp as cddg   # exact rational arithmetic; use `import cdd` for floats

# ---------------------------------------------------------------
# 1. Problem structure: P4 graph A - B - C - D
# ---------------------------------------------------------------
zones = ['A', 'B', 'C', 'D']
dir_edges = [('A','B'), ('B','A'), ('B','C'), ('C','B'), ('C','D'), ('D','C')]

# Dual variable layout (20 vars):
#   x[0:4]   = lam_A, lam_B, lam_C, lam_D    (coverage prices)
#   x[4:8]   = mu_A,  mu_B,  mu_C,  mu_D     (over-procurement cap prices)
#   x[8:14]  = alpha_e, e in dir_edges order (cost shares billed to neighbors)
#   x[14:20] = beta_e,  e in dir_edges order (congestion rents)
NV = 20
lam_ix  = {z: i          for i, z in enumerate(zones)}
mu_ix   = {z: 4 + i      for i, z in enumerate(zones)}
al_ix   = {e: 8 + i      for i, e in enumerate(dir_edges)}
be_ix   = {e: 14 + i     for i, e in enumerate(dir_edges)}

out_edges = {z: [e for e in dir_edges if e[0] == z] for z in zones}

# ---------------------------------------------------------------
# 2. H-representation. cdd row convention: [b, a_1..a_n] means b + a.x >= 0
# ---------------------------------------------------------------
rows = []

# Node constraints (from primal r_z):
#   lam_z + sum_{e out of z} alpha_e - mu_z <= 1
#   ->  1 - lam_z - sum alpha_e + mu_z >= 0
for z in zones:
    a = [Fraction(0)] * NV
    a[lam_ix[z]] = Fraction(-1)
    a[mu_ix[z]]  = Fraction(1)
    for e in out_edges[z]:
        a[al_ix[e]] = Fraction(-1)
    rows.append([Fraction(1)] + a)

# Edge constraints (from primal y_{u,z}):
#   lam_receiver <= alpha_e + beta_e   ->   -lam_recv + alpha_e + beta_e >= 0
for e in dir_edges:
    a = [Fraction(0)] * NV
    a[lam_ix[e[1]]] = Fraction(-1)
    a[al_ix[e]]     = Fraction(1)
    a[be_ix[e]]     = Fraction(1)
    rows.append([Fraction(0)] + a)

# Nonnegativity: x_i >= 0
for i in range(NV):
    a = [Fraction(0)] * NV
    a[i] = Fraction(1)
    rows.append([Fraction(0)] + a)

# ---------------------------------------------------------------
# 3. Enumerate generators
# ---------------------------------------------------------------
mat = cddg.matrix_from_array(rows, rep_type=cddg.RepType.INEQUALITY)
poly = cddg.polyhedron_from_matrix(mat)
gens = cddg.copy_generators(poly)

vertices, rays = [], []
for g in gens.array:
    (vertices if g[0] == 1 else rays).append([Fraction(x) for x in g[1:]])

print(f"raw generators: {len(vertices)} vertices, {len(rays)} rays")

# ---------------------------------------------------------------
# 4. Project to price vectors eta (length 10:  4 D-coeffs then 6 C-coeffs)
#    eta_Dz = lam_z - mu_z ;  eta_Ce = -beta_e   (alphas drop out)
# ---------------------------------------------------------------
def project(x):
    d_part = [x[lam_ix[z]] - x[mu_ix[z]] for z in zones]
    c_part = [-x[be_ix[e]] for e in dir_edges]
    return tuple(d_part + c_part)

etas = sorted({project(v) for v in vertices})   # set() dedupes exactly (rationals)
print(f"distinct price vectors after projection/dedupe: {len(etas)}")

# ---------------------------------------------------------------
# 5. Sanity checks
# ---------------------------------------------------------------
# Anchor 1: "everyone self-covers, all lines saturated":
#   lam=1, beta=1 everywhere, mu=alpha=0  ->  eta = (1,1,1,1, -1,-1,-1,-1,-1,-1)
anchor1 = tuple([Fraction(1)]*4 + [Fraction(-1)]*6)
# Anchor 2: "B fully fed by both neighbors": lam_B=1, alpha_AB=alpha_CB=1
#   ->  eta = (0,1,0,0, 0,0,0,0,0,0)
anchor2 = tuple([Fraction(0), Fraction(1)] + [Fraction(0)]*8)
print("anchor 1 present:", anchor1 in etas)
print("anchor 2 present:", anchor2 in etas)

# Rays must project to nonpositive value on the whole orthant, i.e. every
# coefficient <= 0 (otherwise some instance would have unbounded dual ->
# infeasible primal, impossible here). A violation means a sign error above.
ray_ok = all(all(c <= 0 for c in project(rr)) for rr in rays)
print("all rays project nonpositive:", ray_ok)

# ---------------------------------------------------------------
# 6. Fuzz test: catalog vs direct LP on random instances
# ---------------------------------------------------------------
try:
    import pulp, random
    recv_in = {z: [e for e in dir_edges if e[1] == z] for z in zones}

    def lp_solve(D, C):
        P = pulp.LpProblem('p', pulp.LpMinimize)
        r = {z: pulp.LpVariable(f'r_{z}', lowBound=0) for z in zones}
        y = {e: pulp.LpVariable(f'y_{e[0]}{e[1]}', lowBound=0) for e in dir_edges}
        P += pulp.lpSum(r.values())
        for z in zones:
            P += r[z] + pulp.lpSum(y[e] for e in recv_in[z]) >= D[z]
            P += r[z] <= D[z]
        for e in dir_edges:
            P += y[e] <= r[e[0]]
            P += y[e] <= C[e]
        P.solve(pulp.PULP_CBC_CMD(msg=0))
        return pulp.value(P.objective)

    def catalog_value(D, C):
        theta = [D[z] for z in zones] + [C[e] for e in dir_edges]
        return max(sum(float(c) * t for c, t in zip(eta, theta)) for eta in etas)

    random.seed(0)
    worst = 0.0
    N = 500
    for _ in range(N):
        D = {z: random.uniform(0.1, 10) for z in zones}
        C = {e: random.uniform(0.1, 10) for e in dir_edges}
        worst = max(worst, abs(lp_solve(D, C) - catalog_value(D, C)))
    print(f"fuzz test over {N} random instances: worst |LP - catalog| = {worst:.2e}")
except ImportError:
    print("pulp not installed -- skipping fuzz test (pip install pulp)")

# ---------------------------------------------------------------
# 7. Show the catalog (or its head, if large)
# ---------------------------------------------------------------
labels = [f"D_{z}" for z in zones] + [f"C_{e[0]}{e[1]}" for e in dir_edges]
print("\nfirst 15 price vectors (columns:", ", ".join(labels), "):")
for eta in etas[:15]:
    print("  ", tuple(str(c) for c in eta))