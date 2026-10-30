"""Build uebung/Assignment3_StepResponse.ipynb: Assignment 3 (step response), using Assignments 1 and 2."""
import sys

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# Assignment 3 – Step response: single-storey two-zone office building (Chur)

**Team:** _Valerio Paganini_

Uses the thermal circuit of Assignment 1 ([Assignment1_Model.ipynb](Assignment1_Model.ipynb)) and the state-space
model of Assignment 2 ([Assignment2_SteadyState.ipynb](Assignment2_SteadyState.ipynb)).

Step 1 (free-floating test, as in 03CubicBuilding): $K_p = 0$, all temperatures 0 °C at $t = 0$, then outdoor and
basement at 10 °C. After the settling time every temperature must be 10 °C.

Step 2 (controlled): $K_p = 10^5$ W/K, set-point 21 °C, starting from the design steady state of Assignment 2
($T_o = -7.2$ °C); then $T_o$ jumps to 10 °C. The basement stays at 0.96 °C (assumed slow). This shows how fast the
heating load follows the outdoor temperature."""),
    code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm4bem
from IPython.utils import io

with io.capture_output():                   # model (Assignment 1) and state space (Assignment 2), outputs not shown
    %run Assignment2_SteadyState.ipynb

states = np.flatnonzero(np.diag(C))         # nodes with capacity = state variables, in the order of the nodes
s_air = np.searchsorted(states, [air1, air2])
print(f'{len(states)} state variables, {len(u_names)} inputs, outputs θ{air1}, θ{air2} (air of zone 1 and 2)')"""),
    md("""## 1 Time step and settling time

Explicit Euler is stable if $|1 + \\lambda_i \\Delta t| < 1$ for all eigenvalues $\\lambda_i$ of $A_s$. These are real
and negative, so

$$\\Delta t < \\Delta t_{max} = \\min\\left(-\\frac{2}{\\lambda_i}\\right) = 2 \\min T_i, \\qquad T_i = -\\frac{1}{\\lambda_i}$$

The settling time is about four times the largest time constant. $A_s$ depends on $K_p$, so both models are
evaluated, with the state dominating the fastest and slowest mode (largest eigenvector component)."""),
    code("""models = {'free-floating (K_p = 0)': cases['case 1: test']['ss'],
          'controlled (K_p = 1e5 W/K)': cases['case 2: design heating']['ss']}
tab = {}
for name, (As, Bs, Cs, Ds) in models.items():
    λ, V = np.linalg.eig(As)
    assert np.abs(λ.imag).max() < 1e-12 and λ.real.max() < 0     # real and negative
    λ, V = λ.real, V.real
    fast, slow = λ.argmin(), λ.argmax()
    tab[name] = {'smallest time constant (s)': -1 / λ[fast],
                 'fastest mode, mainly': node_name(states[np.abs(V[:, fast]).argmax()]),
                 'largest time constant (d)': -1 / λ[slow] / 86400,
                 'slowest mode, mainly': node_name(states[np.abs(V[:, slow]).argmax()]),
                 'Δt_max = 2 min T_i (s)': -2 / λ[fast],
                 'settling time 4 max T_i (d)': -4 / λ[slow] / 86400}
tab = pd.DataFrame(tab)
tab"""),
    md("""Each explicit step multiplies a mode by $1 + \\lambda_i \\Delta t$:

| $\\Delta t$ | $1 + \\lambda_i \\Delta t$ | explicit solution |
|---|---|---|
| $< T_{min}$ | 0 to 1 | smooth decay, like the exact solution |
| $T_{min}$ to $2 T_{min} = \\Delta t_{max}$ | −1 to 0 | stable, oscillates between steps |
| $> \\Delta t_{max}$ | below −1 | unstable |

Rounding $\\Delta t_{max}$ down to whole minutes (03CubicBuilding) gives a step close to $\\Delta t_{max}$ and visible
oscillations (end of section 3), so $\\Delta t$ is set below $T_{min}$ instead: 515 s free-floating, rounded down to
whole minutes; 8 s controlled, rounded down to whole seconds. The controller ties the small air capacity to the
set-point, $T = C_{air} / (K_p + \\sum G)$. Each run lasts the settling time rounded up to whole hours."""),
    code("""T_min1, T_min2 = tab.loc['smallest time constant (s)']
dt1 = np.floor(T_min1 / 60) * 60
dt2 = np.floor(T_min2)
dur1, dur2 = (np.ceil(tab.loc['settling time 4 max T_i (d)'] * 24) * 3600).values
print(f'step 1 (free-floating): Δt = {dt1:.0f} s = {dt1 / 60:.0f} min, duration {dur1 / 86400:.1f} d, '
      f'{int(dur1 / dt1)} time steps')
print(f'step 2 (controlled):    Δt = {dt2:.0f} s, duration {dur2 / 86400:.1f} d, {int(dur2 / dt2)} time steps')"""),
    md("""## 2 Input vectors $u$

Order as in Assignment 2: 20 temperature sources $b_T$ (branch order), then 28 flow-rate sources $f_Q$ (node order).
$u$ is constant for $t \\geq 0$.

| | step 1 | step 2, before $t = 0$ | step 2, after $t = 0$ |
|---|---|---|---|
| $K_p$ (W/K) | 0 | $10^5$ | $10^5$ |
| $T_o$ (°C) | 10 | −7.2 | 10 |
| $T_{bsmt}$ (°C) | 10 | 0.96 | 0.96 |
| $T_{sp}$ (°C) | 20 (no effect) | 21 | 21 |
| flow-rate sources (W) | 0 | 0 | 0 |
| initial state $\\theta_C(0)$ | 0 °C | — | steady state before $t = 0$ |"""),
    code("""def u_of(b_v, f_v):
    \"\"\"Input vector u = [b_T, f_Q] from the vectors b and f (order of dm4bem.tc2ss).\"\"\"
    return np.hstack([b_v[b_ptr == 1], f_v[f_ptr == 1]])


u1 = u_of(*inputs(T_o=10, T_bsmt=10, T_sp=20))              # step 1
u2_0 = u_of(*inputs(T_o=T_o2, T_bsmt=T_bsmt2, T_sp=21))     # step 2 before t = 0 (case 2 of Assignment 2)
u2 = u_of(*inputs(T_o=10, T_bsmt=T_bsmt2, T_sp=21))         # step 2 after t = 0
pd.DataFrame({'step 1': u1, 'step 2, t < 0': u2_0, 'step 2, t ≥ 0': u2}, index=u_names).loc[lambda d: d.any(axis=1)]"""),
    md("""## 3 Integration in time

Explicit (Euler forward) and implicit (Euler backward):

$$\\theta_{C,k+1} = (I + \\Delta t A_s)\\,\\theta_{C,k} + \\Delta t B_s u_k \\qquad \\text{(explicit)}$$

$$\\theta_{C,k+1} = (I - \\Delta t A_s)^{-1}(\\theta_{C,k} + \\Delta t B_s u_k) \\qquad \\text{(implicit)}$$

with outputs $y_k = C_s \\theta_{C,k} + D_s u_k$. The implicit method is unconditionally stable and is also run with
$\\Delta t = 1$ h in step 2."""),
    code("""def simulate(ss, u, θ0, dt, duration, method='explicit', keep_states=False):
    \"\"\"Step response with constant input u from the initial state θ0.

    Returns the time t (s), the outputs y (2 × n) and, if keep_states, the states θ (n_s × n).\"\"\"
    As, Bs, Cs, Ds = ss
    n = int(duration / dt) + 1
    I = np.eye(As.shape[0])
    if method == 'explicit':
        M, N = I + dt * As, dt * Bs @ u
    else:
        M = np.linalg.inv(I - dt * As)
        M, N = M, M @ (dt * Bs @ u)
    θ = np.zeros((As.shape[0], n)) if keep_states else None
    y = np.zeros((Cs.shape[0], n))
    θk = θ0.copy()
    for k in range(n):
        y[:, k] = Cs @ θk + Ds @ u
        if keep_states:
            θ[:, k] = θk
        θk = M @ θk + N
    return np.arange(n) * dt, y, θ"""),
    md("""### Step 1 – free-floating building"""),
    code("""ss1 = models['free-floating (K_p = 0)']
θ0 = np.zeros(len(states))
t1, y1_exp, θ1_exp = simulate(ss1, u1, θ0, dt1, dur1, 'explicit', keep_states=True)
_, y1_imp, _ = simulate(ss1, u1, θ0, dt1, dur1, 'implicit')

show = {air1: 'air', 13: 'glass, window S', 11: 'wall S, concrete 4 (inner)', 1: 'wall S, insulation 1 (outer)',
        56: 'roof, concrete 4 (inner)', 69: 'floor, concrete 4 (inner)'}
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(t1 / 86400, y1_exp[0], label='explicit')
axs[0].plot(t1 / 86400, y1_imp[0], ':', label='implicit')
axs[0].set(title=f'Air θ{air1} (zone 1), Δt = {dt1 / 60:.0f} min', xlabel='time, d', ylabel='θ, °C')
for i, lab in show.items():
    axs[1].plot(t1 / 86400, θ1_exp[np.searchsorted(states, i)], label=f'θ{i} {lab}')
axs[1].set(title='Some states of zone 1 (explicit)', xlabel='time, d', xscale='log', xlim=(1e-2, dur1 / 86400))
for ax in axs:
    ax.axhline(10, color='grey', lw=0.8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

t95 = t1[np.argmax(y1_exp[0] >= 9.5)]
print(f'air θ{air1}: 95 % of the step (9.5 °C) after {t95 / 86400:.1f} d; '
      f'at the end {y1_exp[0, -1]:.4f} °C (explicit), {y1_imp[0, -1]:.4f} °C (implicit); steady state (Assignment 2) '
      f'{cases["case 1: test"]["y_ss"][0]:.4f} °C')
print(f'max. difference explicit - implicit: {np.abs(y1_exp - y1_imp).max():.3f} K')"""),
    md("""### Step 2 – controlled building, outdoor temperature step

Initial state $\\theta_C(0) = -A_s^{-1} B_s u(t<0)$ (steady state before the step); heating load
$q_{HVAC} = K_p (T_{sp} - y)$."""),
    code("""ss2 = models['controlled (K_p = 1e5 W/K)']
As, Bs, Cs, Ds = ss2
K_p = 1e5
θ0 = -np.linalg.inv(As) @ Bs @ u2_0                           # steady state before the step
runs = {f'explicit, Δt = {dt2:.0f} s': simulate(ss2, u2, θ0, dt2, dur2, 'explicit'),
        'implicit, Δt = 1 h': simulate(ss2, u2, θ0, 3600, dur2, 'implicit')}
q = {name: K_p * (21 - y[0]) for name, (t, y, _) in runs.items()}   # W, zone 1

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for (name, (t, y, _)), kw in zip(runs.items(), (dict(), dict(ls='', marker='o', ms=3))):
    axs[0].plot(t / 3600, q[name] / 1000, label=name, **kw)
    axs[1].plot(t / 86400, q[name] / 1000, label=name, **kw)
axs[0].set(xlim=(-0.2, 12), title='First 12 hours', xlabel='time, h')
axs[1].set(title='Whole duration', xlabel='time, d')
for ax in axs:
    ax.set(ylabel='q_HVAC zone 1, kW')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

t, y, _ = runs[f'explicit, Δt = {dt2:.0f} s']
qe = q[f'explicit, Δt = {dt2:.0f} s']
q_end = K_p * (21 - (-Cs @ np.linalg.inv(As) @ Bs + Ds) @ u2)[0]   # steady state after the step
for label, tk in [('before the step', 0), ('after 1 time step', dt2), ('after 15 min', 900), ('after 1 h', 3600),
                  ('after 1 d', 86400),
                  ('after 7 d', 7 * 86400), ('at the end', t[-1])]:
    k = int(round(tk / dt2))
    print(f'{label:20s} q_HVAC = {qe[k]:7.0f} W, air θ{air1} = {y[0, k]:.3f} °C')
print(f'steady state after the step: q_HVAC = {q_end:.0f} W; zone 2 equal to zone 1: '
      f'{np.allclose(y[0], y[1])}')
print('share of the total drop of q_HVAC: ' + ', '.join(
    f'{(qe[0] - qe[int(round(tk / dt2))]) / (qe[0] - q_end):.0%} after {label}'
    for label, tk in [('1 time step', dt2), ('15 min', 900), ('1 h', 3600), ('1 d', 86400), ('7 d', 7 * 86400)]))
print(f'ventilation alone: G_v ΔT_o = {G_v * (10 - T_o2):.0f} W = {G_v * (10 - T_o2) / (qe[0] - q_end):.0%} of the drop')"""),
    md("""### Time step of the explicit method

Step 2 with $\\Delta t$ below $T_{min}$ (chosen), between $T_{min}$ and $\\Delta t_{max}$ (whole seconds, as in
03CubicBuilding) and 5 % above $\\Delta t_{max}$. Air temperature for the first four steps and after 1 h:"""),
    code("""dt_max2 = tab.loc['Δt_max = 2 min T_i (s)', 'controlled (K_p = 1e5 W/K)']
for dt in (dt2, np.floor(dt_max2), 1.05 * dt_max2):
    t, y, _ = simulate(ss2, u2, θ0, dt, 3600, 'explicit')
    print(f'Δt = {dt:4.1f} s = {dt / T_min2:.2f} T_min, 1 + λΔt = {1 - dt / T_min2:5.2f}: air '
          f'{", ".join(f"{v:.3f}" for v in y[0, :4])} … {y[0, -1]:.4g} °C after 1 h')"""),
    md("""## 4 Discussion

**Time step.** The fastest mode is the air. Free-floating, $T_{min}$ = 515 s (ventilation and surfaces), so explicit
Euler is stable up to 17 min; 8 min is used. With the controller, $T_{min}$ = 8 s ($C_{air}/(K_p + \\sum G)$), the limit
is 16 s and 7 s is used. At 15 s the solution oscillates, 5 % above $\\Delta t_{max}$ it diverges. Implicit Euler with
1 h gets the slow part of step 2 right but not the first hour.

**Settling time.** The largest time constant, in the floor concrete above the basement, is 17.6 d free-floating and
7.9 d controlled, giving 70 d and 31 d. The controller shortens it by fixing the air temperature.

**Step 1.** All temperatures reach 10 °C, the steady state of Assignment 2 (case 1). The air ends at 9.85 °C (98 %),
with 95 % after 50 d. Outer insulation and glass react within hours, but only to about 7.5 °C and 6 °C; the rest
follows the concrete over days to weeks. The air is coupled more strongly to the concrete (inner surfaces) than to
outdoors (ventilation), so it follows the concrete. Explicit and implicit differ by at most 0.13 K.

**Step 2.** The air stays at 21 °C (within 0.06 K). The heating load falls from 5.5 kW to 2.6 kW. 26 % of the drop
comes in the first time step, mainly from the ventilation, which has no capacity ($G_v \\Delta T_o$ = 0.86 kW, 30 % of
the drop, part of it first going into the surfaces). The windows follow within hours (42 % after 1 h), walls, roof and
floor within days (73 % after 1 d, 98 % after 7 d), since the change must pass insulation and concrete. Both zones
respond identically.

**Limits.** The P-controller is ideal (unlimited power, heating and cooling, no dead-band). Weather response
(Assignment 4) needs time-varying inputs."""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python"}
nbf.write(nb, sys.argv[1])
print("written", sys.argv[1])
