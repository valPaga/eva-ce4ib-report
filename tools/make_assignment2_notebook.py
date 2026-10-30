"""Build uebung/Assignment2_SteadyState.ipynb: Assignment 2 (steady state), using the model of Assignment 1."""
import sys

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# Assignment 2 – Steady-state: single-storey two-zone office building (Chur)

**Team:** _Valerio Paganini_

Intermediate submission of the project (course *dm4bem*, Ch. Ghiaus). This notebook uses the thermal circuit of
Assignment 1 ([Assignment1_Model.ipynb](Assignment1_Model.ipynb), Figure 1 there): 155 temperature nodes,
191 branches, two zones (south and north half) coupled by an interior wall and a closed door.

Content, as in the list of the assignment:

1. matrices $A$, $G$ and $C$ of the system of differential-algebraic equations (DAE) in Python,
2. values of the inputs (temperature sources $b$ and flow-rate sources $f$),
3. steady-state response of the DAE,
4. matrices $A_s$, $B_s$, $C_s$, $D_s$ of the state-space representation,
5. input vector $u = [b_T^T, f_Q^T]^T$,
6. steady-state response of the state-space representation,
7. comparison of the two results,
8. visual check of the matrices (addition)."""),
    code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm4bem
from IPython.utils import io"""),
    md("""## 1 Matrices $A$, $G$ and $C$ of the DAE

The matrices are written in Python in Assignment 1. That notebook is run here without showing its outputs; it
defines the incidence matrix `A` (191 × 155), the diagonal matrices `G` (191 × 191, W/K) and `C` (155 × 155, J/K),
the vectors of pointers `b`, `f`, `y` and the list `branches` (from node, to node, conductance, temperature source)."""),
    code("""with io.capture_output():                   # model of Assignment 1, its outputs are not shown
    %run Assignment1_Model.ipynb

b_ptr, f_ptr, y_ptr = b.copy(), f.copy(), y.copy()     # pointers: 1 = temperature source / flow source / output
src = pd.Series({k: s for k, (_, _, _, s) in enumerate(branches) if s})   # source of every branch with b = 1
k_ctrl = list(src.index[src.str.startswith('T_sp')])                      # P-controllers: q88, q177

print(f'A {A.shape}, G {G.shape}, C {C.shape}')
print(f'{int(b_ptr.sum())} temperature sources, {int(f_ptr.sum())} flow-rate sources, outputs θ{air1}, θ{air2}')
print(f'P-controllers on branches {[f"q{k}" for k in k_ctrl]}')"""),
    md("""## 2 Inputs: temperature sources $b$ and flow-rate sources $f$

Three cases are calculated.

**Case 1 – test of the model** (as in 03CubicBuilding): controller off ($K_p = 0$), outdoor and basement temperature
$T_o = T_{bsmt} = 10$ °C, set-point $T_{sp} = 20$ °C (without effect when $K_p = 0$), all flow-rate sources zero. In
steady state all temperatures must then be 10 °C. This is a test that can show that the model is wrong
(falsification); passing it does not prove that the model is right.

**Case 2 – design heating load**: design outdoor temperature of the SIA climate station Chur $T_o = -7.2$ °C
(assumption), basement $T_{bsmt} = 20 - 0.7\\,(20 - T_o) = 0.96$ °C (hypothesis 5 of Assignment 1, with
$T_{o,24h} = T_o$ in steady state), set-point $T_{sp} = 21$ °C in both zones, P-controller active with
$K_p = 10^5$ W/K, no solar radiation and no internal gains.

**Case 3 – zone 1 occupied, zone 2 in set-back**: as case 2, but with the internal gains of zone 1 at full load
(hypothesis 9; SIA 2024 room type 3.01, assumption): occupants 84 W per person at 14 m² per person, equipment
11 W/m², lighting 9.72 W/m², together 6.0 kW on the air node $\\theta_{71}$. Zone 2 is unoccupied, with the set-back
set-point $T_{sp,2} = 18$ °C (hypothesis 12). This case has a flow-rate source and different set-points in the two
zones.

The functions below give the vectors $b$ (°C, one value per branch) and $f$ (W, one value per node) and the
conductance matrix with the chosen gain of the P-controllers."""),
    code("""def inputs(T_o, T_bsmt, T_sp, Q=None):
    \"\"\"Values of the temperature sources b (°C) and of the flow-rate sources f (W).

    T_sp: one set-point for both zones or a pair (zone 1, zone 2); Q: {node: flow rate in W}.\"\"\"
    T_sp1, T_sp2 = (T_sp, T_sp) if np.isscalar(T_sp) else T_sp
    value = {'T_o': T_o, 'T_bsmt': T_bsmt, 'T_sp,1': T_sp1, 'T_sp,2': T_sp2}
    b = np.zeros(nq)
    for k, s in src.items():
        b[k] = value[s]
    f = np.zeros(nθ)
    for node, Q_node in (Q or {}).items():
        f[node] = Q_node
    return b, f


def conductances(K_p):
    \"\"\"Conductance matrix G of Assignment 1 with the gain K_p (W/K) on the controller branches.\"\"\"
    G_K = G.copy()
    G_K[k_ctrl, k_ctrl] = K_p
    return G_K


T_o2 = -7.2                                 # °C, design outdoor temperature (cases 2 and 3)
T_bsmt2 = 20 - 0.7 * (20 - T_o2)            # °C, basement (hypothesis 5)
Q_occ = S_plan / 14 * 84                    # W, occupants: 14 m²/P, 84 W/P (full load, case 3)
Q_equip = S_plan * 11                       # W, equipment 11 W/m²
Q_light = S_plan * 9.72                     # W, lighting 9.72 W/m²
cases = {
    'case 1: test': dict(G=conductances(0), inputs=inputs(T_o=10, T_bsmt=10, T_sp=20)),
    'case 2: design heating': dict(G=conductances(1e5), inputs=inputs(T_o=T_o2, T_bsmt=T_bsmt2, T_sp=21)),
    'case 3: zone 2 set-back': dict(G=conductances(1e5),
                                    inputs=inputs(T_o=T_o2, T_bsmt=T_bsmt2, T_sp=(21, 18),
                                                  Q={air1: Q_occ + Q_equip + Q_light})),
}
pd.DataFrame({name: {**{f'{s} (°C)': c['inputs'][0][src.index[src == s][0]] for s in src.unique()},
                     'K_p (W/K)': c['G'][k_ctrl[0], k_ctrl[0]],
                     'sum of the flow-rate sources (W)': c['inputs'][1].sum()}
              for name, c in cases.items()})"""),
    md("""## 3 Steady-state response of the DAE

In steady state $C \\dot\\theta = 0$, so

$$\\theta_{ss} = (A^T G A)^{-1}(A^T G b + f), \\qquad q_{ss} = G(-A\\theta_{ss} + b)$$"""),
    code("""for name, c in cases.items():
    Gc = c['G']
    b_v, f_v = c['inputs']
    c['θ'] = np.linalg.inv(A.T @ Gc @ A) @ (A.T @ Gc @ b_v + f_v)
    c['q'] = Gc @ (-A @ c['θ'] + b_v)
    θ = c['θ']
    print(f'{name}: θ min / max = {θ.min():.4f} / {θ.max():.4f} °C, '
          f'air θ{air1} = {θ[air1]:.4f} °C, θ{air2} = {θ[air2]:.4f} °C')"""),
    md("""Case 1 gives 10.00 °C in every node: the model passes the test.

For case 2, the temperatures through the south wall of zone 1 (nodes $\\theta_0 \\dots \\theta_{12}$) show that almost
all the temperature difference is in the insulation:"""),
    code("""(m1, w1), (m2, w2) = wall                                          # layers of the external wall (Assignment 1)
x = np.r_[np.arange(5) * w1 / 4, w1 + np.arange(1, 9) * w2 / 8]    # m, position of θ0 … θ12 from the outside
θ = cases['case 2: design heating']['θ']
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.plot(x, θ[:13], 'o-', label='south wall, zone 1')
ax.axvspan(0, w1, color='gold', alpha=0.25, label=m1.replace('_', ' '))
ax.axvspan(w1, w1 + w2, color='grey', alpha=0.25, label=m2)
ax.axhline(θ[air1], color='tab:red', ls='--', label=f'air θ{air1}')
ax.set(xlabel='x from the outer surface, m', ylabel='θ, °C', title='Case 2: steady-state temperatures in the wall')
ax.legend(fontsize=8)
plt.show()"""),
    md("""Heat losses of case 2: the flow rate in the branch of a temperature source is the heat that flows from that
source into the building, so $-q$ is the heat loss through the element. In steady state without gains, the sum of
the losses equals the heat supplied by the two P-controllers (heating load)."""),
    code("""q = cases['case 2: design heating']['q']
elements = {'wall S/N': 0, 'window S/N': 15, 'wall E': 19, 'window E': 34, 'wall W': 38, 'window W': 53,
            'roof': 57, 'floor (basement)': 72, 'ventilation': 87}
loss = pd.DataFrame({'zone 1 (W)': {e: -q[k] for e, k in elements.items()},
                     'zone 2 (W)': {e: -q[k + 89] for e, k in elements.items()}})
loss.loc['sum of the losses'] = loss.sum()
loss.loc['heating load q_HVAC'] = [q[k_ctrl[0]], q[k_ctrl[1]]]
loss['total (W)'] = loss.sum(axis=1)
share = loss.loc[list(elements), 'total (W)'] / loss.loc['sum of the losses', 'total (W)']
print(f'heating load {q[k_ctrl[0]] / 1000:.2f} kW per zone = {q[k_ctrl[0]] / S_plan:.1f} W/m² of floor; '
      f'air {cases["case 2: design heating"]["θ"][air1]:.2f} °C (set-point 21 °C)')
print(f'shares: windows {share.filter(like="window").sum():.0%}, ventilation {share["ventilation"]:.0%}, '
      f'roof {share["roof"]:.0%}, floor {share["floor (basement)"]:.0%}, walls {share.filter(like="wall").sum():.0%}')
print(f'between the zones: conduction through the interior wall (q179–q186) max |q| = '
      f'{np.abs(q[179:187]).max():.1e} W, door q190 = {abs(q[190]):.1e} W')
θ = cases['case 2: design heating']['θ']
print(f'long-wave star node θ{rm1} = {θ[rm1]:.2f} °C, operative temperature ≈ (θ{air1} + θ{rm1}) / 2 = '
      f'{(θ[air1] + θ[rm1]) / 2:.2f} °C')
loss.round(0)"""),
    md("""Case 3: energy balance of each zone. In steady state the conduction through the interior wall is the same in
all its branches ($q_{179} = \\dots = q_{186}$); together with the door $q_{190}$ it is the heat that zone 1 gives
to zone 2."""),
    code("""q, θ = cases['case 3: zone 2 set-back']['q'], cases['case 3: zone 2 set-back']['θ']
loss1 = -sum(q[k] for k in elements.values())           # zone 1 to outdoor and basement
loss2 = -sum(q[k + 89] for k in elements.values())      # zone 2 to outdoor and basement
q12 = q[179] + q[190]                                   # zone 1 to zone 2: interior wall + door
print(f'air θ{air1} = {θ[air1]:.2f} °C (zone 1), θ{air2} = {θ[air2]:.2f} °C (zone 2)')
print(f'zone 1: internal gains {Q_occ + Q_equip + Q_light:.0f} W + heating {q[k_ctrl[0]]:.0f} W '
      f'= losses {loss1:.0f} W + to zone 2 {q12:.0f} W')
print(f'zone 2: heating {q[k_ctrl[1]]:.0f} W + from zone 1 {q12:.0f} W = losses {loss2:.0f} W')
print(f'zone 1 to zone 2: interior wall {q[179]:.0f} W (q179 … q186), door {q[190]:.1f} W (q190)')"""),
    md("""## 4 State-space representation

`dm4bem.tc2ss` eliminates the nodes without capacity. It needs the pointers $b$, $f$, $y$ (1 = source or output),
not the values. Because $G$ contains $K_p$, the state-space matrices are computed for each case."""),
    code("""for name, c in cases.items():
    c['ss'] = dm4bem.tc2ss(A, c['G'], C, b_ptr, f_ptr, y_ptr)
As, Bs, Cs, Ds = cases['case 2: design heating']['ss']
print(f'As {As.shape}: {As.shape[0]} states = nodes with capacity')
print(f'Bs {Bs.shape}: {Bs.shape[1]} inputs = {int(b_ptr.sum())} temperature sources + {int(f_ptr.sum())} flow-rate sources')
print(f'Cs {Cs.shape}, Ds {Ds.shape}: {Cs.shape[0]} outputs (air temperatures)')"""),
    md("""## 5 Input vector $u = [b_T^T, f_Q^T]^T$

$b_T$ holds the values of $b$ on the branches with a temperature source (pointer $b_k = 1$), in the order of the
branches; $f_Q$ the values of $f$ on the nodes with a flow-rate source (pointer $f_i = 1$), in the order of the nodes.
This is the order of the inputs in `dm4bem.tc2ss`. A source can have the value zero: in cases 1 and 2 all flow-rate
sources are zero, in case 3 only the internal gains of zone 1 on $\\theta_{71}$ are not zero."""),
    code("""u_names = [f'b: q{k} ({src[k]})' for k in np.flatnonzero(b_ptr)] + \\
          [f'f: θ{i} ({"+".join(flow[i])})' for i in np.flatnonzero(f_ptr)]
for name, c in cases.items():
    b_v, f_v = c['inputs']
    c['u'] = np.hstack([b_v[b_ptr == 1], f_v[f_ptr == 1]])
pd.DataFrame({name: c['u'] for name, c in cases.items()}, index=u_names)"""),
    md("""## 6 Steady-state response of the state-space representation

In steady state $\\dot\\theta_C = 0$:

$$y_{ss} = (-C_s A_s^{-1} B_s + D_s)\\, u$$"""),
    code("""for name, c in cases.items():
    As, Bs, Cs, Ds = c['ss']
    c['y_ss'] = (-Cs @ np.linalg.inv(As) @ Bs + Ds) @ c['u']
    print(f'{name}: y_ss = {np.round(c["y_ss"], 4)} °C')"""),
    md("""## 7 Comparison: DAE and state-space representation"""),
    code("""for name, c in cases.items():
    err = np.max(np.abs(c['θ'][y_ptr == 1] - c['y_ss']))
    print(f'{name}: DAE θ{air1} = {c["θ"][air1]:.6f} °C, θ{air2} = {c["θ"][air2]:.6f} °C | '
          f'state space y_ss = {c["y_ss"][0]:.6f}, {c["y_ss"][1]:.6f} °C | max. error {err:.1e} °C')"""),
    md("""The air temperatures from the DAE and from the state-space representation agree to the numerical precision
(error below $10^{-12}$ °C) in all three cases. Case 3 has a flow-rate source and different set-points in the two
zones, so the comparison also checks the order of the flow-rate sources in $u$ and the coupling of the zones; only
sources with the same value (e.g. two $T_o$, or two flow-rate sources equal to zero) could be exchanged unnoticed.

**Discussion**

- Case 1 passes the test: with $T_o = T_{bsmt} = 10$ °C, no flow-rate sources and the controller off ($K_p = 0$),
  every node is at 10 °C.
- Case 2: with $K_p = 10^5$ W/K the air temperature stays 0.06 K below the set-point of 21 °C: the P-controller needs
  a small error to deliver the heat, $T_{sp} - \\theta_a = q_{HVAC}/K_p$. The design heating load is 5.5 kW per zone,
  24.6 W/m² of floor. The windows cause 40 % of the losses, the ventilation 25 %, the roof 16 %, the floor above the
  basement 13 % and the insulated walls 5 %.
- Case 2: the long-wave star node is at 18.8 °C, 2.2 K below the air, because the P-controller heats only the air.
  A U-value assumes surroundings at air temperature on the inner side; here the radiant surroundings are colder, so
  the transmission losses are about 5 % smaller than $U S\\,\\Delta T$ with the air temperature (floor 6 % with $U$
  after EN ISO 6946, 7 % with $U$ of the circuit). The operative temperature is about 19.9 °C.
- Case 2: both zones have the same geometry and, without solar radiation, the same temperatures: no heat flows
  through the interior wall or the door. Inside each zone, the interior wall still exchanges heat with the air and,
  by long-wave radiation, with the colder windows.
- Case 3: the internal gains of 6.0 kW (26.7 W/m²) are larger than the losses of zone 1 to outdoor and basement
  (5.5 kW). Zone 1 still needs about 0.1 kW of heating at the design outdoor temperature, because it gives about
  0.6 kW to zone 2 (18 °C), almost all by conduction through the interior wall, only about 11 W through the closed
  door. Zone 2 needs 4.4 kW."""),
    md("""## 8 Visual check of the matrices

Every non-zero entry is a dot (case 2, $K_p = 10^5$ W/K). The dashed lines separate zone 1, zone 2 and the
interior wall (nodes $\\theta_0, \\theta_{73}, \\theta_{146}$; branches $q_0, q_{89}, q_{178}$) and, in $B_s$, the
temperature sources $b_T$ from the flow-rate sources $f_Q$.

- $A$: each branch (row) has a $-1$ at the node it leaves and a $+1$ at the node it enters; a branch from a
  temperature source has only the $+1$. The layers of the walls give the staircases.
- $A_s$: the states are the 72 nodes with capacity, in the order of the nodes. Neighbouring meshes give the band
  along the diagonal; the eliminated surfaces and star nodes couple the inner meshes, the glass and the air of a zone
  with each other (blocks). Zone 1 and zone 2 are coupled only through the interior wall and the door.
- $B_s$: which source acts on which state, e.g. $T_o$ on the outer insulation mesh, $T_{sp}$ and the internal gains
  on the air."""),
    code("""As, Bs, Cs, Ds = cases['case 2: design heating']['ss']
states = np.flatnonzero(np.diag(C))                 # nodes with capacity = states, in the order of the nodes
n_bT = int(b_ptr.sum())                             # number of temperature sources in u
fig, axs = plt.subplots(1, 3, figsize=(13, 5), gridspec_kw={'width_ratios': [155 / 191, 1, 48 / 72]})   # same height
for ax, M, title in zip(axs, (A, As, Bs), (f'A {A.shape}', f'As {As.shape}', f'Bs {Bs.shape}')):
    ax.spy(M, markersize=1.5)
    ax.set_title(title)
for n in (73, 146):                                 # zones in A: nodes (columns) and branches (rows)
    axs[0].axvline(n - 0.5, color='grey', ls='--', lw=0.8)
for k in (89, 178):
    axs[0].axhline(k - 0.5, color='grey', ls='--', lw=0.8)
for s in np.searchsorted(states, (73, 146)):        # zones in As and Bs: states
    for ax in axs[1:]:
        ax.axhline(s - 0.5, color='grey', ls='--', lw=0.8)
    axs[1].axvline(s - 0.5, color='grey', ls='--', lw=0.8)
axs[2].axvline(n_bT - 0.5, color='tab:red', ls='--', lw=0.8)
axs[0].set(xlabel='node θ', ylabel='branch q')
axs[1].set(xlabel='state', ylabel='state')
axs[2].set(xlabel=f'input u: {n_bT} b_T | {Bs.shape[1] - n_bT} f_Q', ylabel='state')
plt.tight_layout()
plt.show()

# uniform temperature, no flow-rate sources: steady state, so every row of [As, Bs_T] sums to zero
print(f'max |row sum of [As, Bs_T]| = {np.abs(As.sum(axis=1) + Bs[:, :n_bT].sum(axis=1)).max():.1e} 1/s')
for name in ('case 1: test', 'case 2: design heating'):
    lam = np.linalg.eigvals(cases[name]['ss'][0]).real
    print(f'{name}: time constants -1/λ of As from {-1 / lam.min():.0f} s to {-1 / lam.max() / 86400:.1f} d, '
          f'explicit Euler stable for Δt < 2/|λ|max = {2 / -lam.min():.0f} s')"""),
    md("""The row of the air node $\\theta_{71}$ (zone 1) written out. Multiplied by the capacity of the air, the
entries are conductances in W/K: the state equation of the air is the energy balance of the air node, with the
surfaces eliminated. The diagonal entry is minus the sum of all conductances of the air node, so the row sums to
zero (temperature sources included). The air is coupled directly to the inner concrete mesh of each element, to the
glass, to the first mesh of the interior wall and, through the door, to the air of zone 2; the outdoor temperature
acts directly only through the ventilation. The internal gains enter with $B_s \\cdot C_{air} = 1$: the whole flow
rate goes into the air. Of the solar radiation $\\Phi_i$ absorbed on an inner surface only a small part (0.04 to 0.5)
goes directly into the air; the rest goes into the inner mesh behind the surface and, by long-wave radiation, to the
meshes and the glass of the other elements."""),
    code("""def node_name(i):
    \"\"\"Element and layer of node θi (numbering of Assignment 1).\"\"\"
    if i >= 146:                                                   # interior wall θ146 … θ154
        return f'interior wall, concrete {(i - 145) // 2}' if (i - 146) % 2 else f'interior wall θ{i}'
    z, k = divmod(i, 73)
    if k >= 71:
        return f'{("air", "star node")[k - 71]} zone {z + 1}'
    o = 'SN'[z]
    for e, layers, n in [(f'wall {o}', wall, 13), (f'window {o}', None, 2), ('wall E', wall, 13), ('window E', None, 2),
                         ('wall W', wall, 13), ('window W', None, 2), ('roof', roof, 13), ('floor', floor, 13)]:
        if k < n:
            break
        k -= n
    if layers is None:
        return f'{e} zone {z + 1}, {("glass", "inner surface")[k]}'
    mesh = {1: (0, 1), 3: (0, 2), 5: (1, 1), 7: (1, 2), 9: (1, 3), 11: (1, 4)}
    if k in mesh:
        j, m = mesh[k]
        return f'{e} zone {z + 1}, {layers[j][0].replace("_", " ")} {m}'
    return f'{e} zone {z + 1}, {("outer surface" if k == 0 else "inner surface" if k == 12 else f"θ{i}")}'


s_air = int(np.searchsorted(states, air1))          # state of the air node θ71
C_air = C[air1, air1]
row_As = pd.DataFrame({'node': [node_name(i) for i in states], 'As (1/s)': As[s_air],
                       'As · C_air (W/K)': As[s_air] * C_air},
                      index=[f'θ{i}' for i in states])
row_Bs = pd.DataFrame({'Bs': Bs[s_air], 'Bs · C_air': Bs[s_air] * C_air,
                       'unit': ['1/s | W/K'] * n_bT + ['K/J | –'] * (Bs.shape[1] - n_bT)}, index=u_names)
print(f'air node θ{air1} = state {s_air}, C_air = {C_air:.4g} J/K, time constant C_air / Σ G = '
      f'{-1 / As[s_air, s_air]:.0f} s')
print(f'non-zero: {np.count_nonzero(row_As["As (1/s)"])} of {As.shape[1]} in As, '
      f'{np.count_nonzero(row_Bs["Bs"])} of {Bs.shape[1]} in Bs')
print(f'Cs: one 1 per row at states {[int(np.flatnonzero(r)[0]) for r in Cs]} (air nodes θ{air1}, θ{air2}); '
      f'Ds = 0: {not Ds.any()}')
display(row_As[row_As['As (1/s)'] != 0].style.format({'As (1/s)': '{:.3e}', 'As · C_air (W/K)': '{:.1f}'}))
display(row_Bs[row_Bs['Bs'] != 0].style.format({'Bs': '{:.3e}', 'Bs · C_air': '{:.4g}'}))"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python"}
nbf.write(nb, sys.argv[1])
print("written", sys.argv[1])
