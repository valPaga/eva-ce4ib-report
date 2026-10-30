"""Build uebung/Assignment4_Weather.ipynb: Assignment 4 (response to weather), using Assignments 1 and 2."""
import sys

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# Assignment 4 – Response to weather: single-storey two-zone office building (Chur)

**Team:** _Valerio Paganini_

One year of Chur weather on the building of Assignment 1 ([Assignment1_Model.ipynb](Assignment1_Model.ipynb)),
with $A$, $G$, $C$ and the state space of Assignment 2 ([Assignment2_SteadyState.ipynb](Assignment2_SteadyState.ipynb))
and the time step from Assignment 3 ([Assignment3_StepResponse.ipynb](Assignment3_StepResponse.ipynb))."""),
    code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import dm4bem
from IPython.utils import io

with io.capture_output():                   # model (Assignment 1) and state space (Assignment 2), outputs not shown
    %run Assignment2_SteadyState.ipynb

states = np.flatnonzero(np.diag(C))         # nodes with capacity = state variables
bT, fQ = np.flatnonzero(b_ptr), np.flatnonzero(f_ptr)
jb = {k: j for j, k in enumerate(bT)}                   # branch with temperature source -> column of u
jf = {i: len(bT) + j for j, i in enumerate(fQ)}         # node with flow-rate source -> column of u
print(f'{len(states)} state variables, {len(bT)} + {len(fQ)} = {len(u_names)} inputs, outputs θ{air1}, θ{air2}')"""),
    md("""## 1 Start and end time

Typical weather year of Chur (TMYx 2004–2018, hypothesis 4) on the time axis of 2023 (starts on a Sunday). The
concrete has time constants of up to 18 days (Assignment 3), so a 30-day warm-up with the last 30 days of the
year runs first (assumption) and is excluded from the results.

Time step 15 min, implicit Euler; explicit Euler would need $\\Delta t$ < 16 s with the P-controller
(Assignment 3). The step size is checked in section 5."""),
    code("""year, dt = 2023, 900                        # time axis; s, time step (implicit Euler)
start, end = f'{year}-01-01 00:00', f'{year}-12-31 23:45'
warmup = 30                                  # d, warm-up before the start (assumption)
print(f'{start} to {end}, Δt = {dt / 60:.0f} min, warm-up {warmup} d')"""),
    md("""## 2 Inputs

### Weather data and solar irradiance on the walls

Irradiance per surface from `dm4bem.read_epw` and `dm4bem.sol_rad_tilt_surf` (azimuth 0 = south, west positive;
albedo 0.2, assumption), with two corrections. First, `read_epw` labels hourly values with the start of the hour
but `sol_rad_tilt_surf` puts the sun at the end (`minute + 60`, clock time). The middle of the hour in solar time
is $0.5\\,\\text{h} + (15° \\cdot 1 - 9.53°)/15°\\,\\text{h} = 0.86$ h (52 min) earlier for Chur (UTC+1, 9.53° E;
equation of time neglected), so the time is shifted by that; otherwise east would get 37 % less and west 35 % more
direct radiation. Second, the ground-reflected part uses $1 - \\cos\\beta / 2$ instead of the view factor
$(1 - \\cos\\beta)/2$ (Duffie & Beckman), which doubles it on the façades, so it is corrected."""),
    code("""weather, meta = dm4bem.read_epw('daten/CHE_GR_Chur.067860_TMYx.2004-2018.epw', coerce_year=year)
weather = weather[['temp_air', 'dir_n_rad', 'dif_h_rad']]
albedo = 0.2
surfaces = {'S': (90, 0), 'E': (90, -90), 'W': (90, 90), 'N': (90, 180), 'roof': (0, 0)}   # slope, azimuth
shift = pd.Timedelta(hours=0.5 + (15 * meta['TZ'] - meta['longitude']) / 15)   # see the first point above
w_sun = weather.copy()
w_sun.index = w_sun.index - shift
E = {}
for o, (slope, azimuth) in surfaces.items():
    rad = dm4bem.sol_rad_tilt_surf(w_sun, {'slope': slope, 'azimuth': azimuth, 'latitude': meta['latitude']},
                                   albedo)
    rad.index = weather.index
    cb = np.cos(np.radians(slope))
    rad['reflected'] *= ((1 - cb) / 2) / (1 - cb / 2)         # view factor to the ground (see above)
    E[o] = rad.sum(axis=1)
weather = pd.DataFrame({'T_o': weather['temp_air'], **{f'E_{o}': e for o, e in E.items()}})
weather.index = weather.index.tz_localize(None)
print(f'latitude {meta["latitude"]}°, T_o mean {weather["T_o"].mean():.1f} °C, '
      f'min {weather["T_o"].min():.1f} °C, max {weather["T_o"].max():.1f} °C')
(weather.filter(like='E_').sum() / 1000).round(0).rename('solar irradiation, kWh/(m²·a)').to_frame().T"""),
    md("""### Resampling and warm-up

Linear interpolation to the time step; the last 30 days are put in front as warm-up."""),
    code("""pre = weather.loc[weather.index[-1] - pd.Timedelta(days=warmup):].iloc[1:].copy()
pre.index = pre.index - pd.Timedelta(days=365)                 # last 30 days, one year earlier
w = pd.concat([pre, weather]).resample(f'{dt}s').interpolate()
w = w.loc[:end]
t = w.index
n = len(t)
k0 = t.get_loc(pd.Timestamp(start))                            # first time step of the year
print(f'{n} time steps ({k0} of warm-up), {t[0]} to {t[-1]}')"""),
    md("""### Other inputs

Internal gains (hypothesis 9; SIA 2024 room type 3.01, assumption), all convective on the air node: occupancy
Mon–Fri 07:00–18:00; occupants 84 W per person at 14 m² per person and equipment 11 W/m², with hourly profiles and
an annual factor of 0.8 (holidays, absence); equipment at least 30 % (stand-by), weekends included; lighting
9.72 W/m² during occupancy, scaled to 13.45 kWh/m² per year.

Set-points (hypothesis 12): heating/cooling 21/26 °C during occupancy, 18/28 °C otherwise.
Basement (hypothesis 5): $T_{bsmt} = 20 - 0.7\\,(20 - T_{o,24h})$, $T_{o,24h}$ = 24-hour mean of $T_o$.

Solar radiation (hypothesis 8; values assumed):

| Quantity | Value |
|---|---|
| absorptivity façade / roof | 0.5 / 0.6 |
| glazing: transmitted / absorbed | 0.40 / 0.20 (g ≈ 0.5 incl. absorbed part going inside) |
| horizon, reveals, angle of incidence $F_{sol}$ | 0.65 |
| frame (no solar gain) | 20 % of the window |
| external blinds closed: factor on transmitted and absorbed | 0.20 ($g_{tot} \\approx 0.10$) |
| blinds closed if | façade irradiance > 200 W/m² and 24-hour mean $T_o$ > 12 °C |
| transmitted radiation | 50 % on the floor, 50 % on walls, roof and interior wall by area (not windows) |

The blind rule depends only on the weather, so all inputs are known in advance (also for the MPC)."""),
    code("""p_persons = np.array([0, 0, 0, 0, 0, 0, 0, 0.2, 0.6, 1, 1, 0.8, 0.4, 0.6, 1, 0.8, 0.6, 0.2, 0, 0, 0, 0, 0, 0])
p_equip = np.array([0.3] * 8 + [0.6, 0.8, 1, 0.8, 0.4, 0.6, 1, 0.8, 0.6] + [0.3] * 7)
f_year, f_standby = 0.8, 0.3                                     # annual factor, stand-by share of equipment

workday = t.dayofweek < 5
h = t.hour + t.minute / 60
occupied = workday & (h >= 7) & (h < 18)
q_pers = np.where(workday, p_persons[t.hour] * f_year, 0) * 84 / 14          # W/m²
q_equip = np.where(workday, np.maximum(p_equip[t.hour] * f_year, f_standby), f_standby) * 11
f_light = 13.45e3 / (9.72 * occupied[k0:].sum() * dt / 3600)                  # annual energy 13.45 kWh/m²
q_light = np.where(occupied, f_light, 0) * 9.72
q_int = q_pers + q_equip + q_light                                            # W/m² of floor

T_h = np.where(occupied, 21.0, 18.0)                             # °C, heating set-point
T_c = np.where(occupied, 26.0, 28.0)                             # °C, cooling set-point
T_o24 = w['T_o'].rolling('24h', min_periods=1).mean().to_numpy()
T_bsmt = 20 - 0.7 * (20 - T_o24)

alpha_wall, alpha_roof = 0.5, 0.6
tau_g, alpha_g, F_sol, F_c = 0.40, 0.20, 0.65, 0.20
blind = {o: (w[f'E_{o}'].to_numpy() > 200) & (T_o24 > 12) for o in ('S', 'E', 'W', 'N')}
print(f'internal gains: mean {q_int[k0:].mean():.1f} W/m², max {q_int.max():.1f} W/m², '
      f'{q_int[k0:].sum() * dt / 3.6e6:.1f} kWh/(m²·a); lighting factor {f_light:.2f}')
print('blinds closed (h/a): ' + ', '.join(f'{o} {blind[o][k0:].sum() * dt / 3600:.0f}' for o in blind))"""),
    md("""### Input vector in time

One row per time step, 48 columns as in Assignment 2: temperature sources $b_T$ (branches), then flow-rate
sources $f_Q$ (nodes). Nodes of zone 1 (zone 2: + 73): outer surfaces
$\\theta_0, \\theta_{15}, \\theta_{30}$ (walls S, E, W), $\\theta_{45}$ (roof); glass $\\theta_{13}, \\theta_{28},
\\theta_{43}$; inner surfaces $\\theta_{12}, \\theta_{27}, \\theta_{42}$ (walls), $\\theta_{57}$ (roof),
$\\theta_{70}$ (floor); interior wall $\\theta_{146}$ (zone 1 side), $\\theta_{154}$ (zone 2 side); air
$\\theta_{71}$. The controller sets $T_{sp}$ in the time loop."""),
    code("""U = np.zeros((n, len(u_names)))
for k in bT:                                                     # temperature sources
    U[:, jb[k]] = {'T_o': w['T_o'].to_numpy(), 'T_bsmt': T_bsmt}.get(src[k], 0.0)
j_sp = [jb[k] for k in k_ctrl]                                   # columns of T_sp,1 and T_sp,2

Q_sol = {}                                                       # W, transmitted solar radiation per zone
for z, n0, o1, iw in ((1, 0, 'S', 146), (2, 73, 'N', 154)):
    Q_t = np.zeros(n)
    for o, size, k_out, k_g in ((o1, 'long', 0, 13), ('E', 'short', 15, 28), ('W', 'short', 30, 43)):
        E_o = w[f'E_{o}'].to_numpy()
        S_g = S_win[size] * (1 - frame)                          # glass area
        shade = np.where(blind[o], F_c, 1.0)
        U[:, jf[n0 + k_out]] += alpha_wall * S_wall[size] * E_o                 # Φo, outer wall surface
        U[:, jf[n0 + k_g]] += alpha_g * F_sol * shade * S_g * E_o               # Φa, glass
        Q_t += tau_g * F_sol * shade * S_g * E_o
    U[:, jf[n0 + 45]] += alpha_roof * S_plan * w['E_roof'].to_numpy()           # Φo, roof
    inner = {n0 + 12: S_wall['long'], n0 + 27: S_wall['short'], n0 + 42: S_wall['short'],
             n0 + 57: S_plan, iw: S_iw}                                         # inner surfaces except the floor
    U[:, jf[n0 + 70]] += 0.5 * Q_t                                              # Φi, floor
    for node, S in inner.items():
        U[:, jf[node]] += 0.5 * Q_t * S / sum(inner.values())                   # Φi, other inner surfaces
    U[:, jf[n0 + 71]] += S_plan * q_int                                         # internal gains, air
    Q_sol[z] = Q_t
U = pd.DataFrame(U, index=t, columns=u_names)
U.loc[f'{year}-07-17':f'{year}-07-17 23:45'].iloc[::8].loc[:, lambda d: d.abs().sum() > 0].round(0).T"""),
    code("""wk = slice(f'{year}-07-17', f'{year}-07-23')                     # a summer week (Monday to Sunday)
fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
w.loc[wk, 'T_o'].plot(ax=axs[0], label='T_o')
axs[0].plot(t[(t >= '2023-07-17') & (t < '2023-07-24')], T_bsmt[(t >= '2023-07-17') & (t < '2023-07-24')],
            label='T_bsmt')
w.loc[wk].filter(like='E_').plot(ax=axs[1])
U.loc[wk, f'f: θ{air1} (Q̇occ + Q̇equip + Q̇light)'].plot(ax=axs[2], label='internal gains zone 1')
pd.Series(Q_sol[1], t).loc[wk].plot(ax=axs[2], label='transmitted solar zone 1 (blinds)')
pd.Series(Q_sol[2], t).loc[wk].plot(ax=axs[2], label='transmitted solar zone 2 (blinds)')
axs[0].set(ylabel='°C', title='Inputs in a summer week')
axs[1].set(ylabel='irradiance, W/m²')
axs[2].set(ylabel='W')
for ax in axs:
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()"""),
    md("""## 3 Initial conditions

All states start at 20 °C at the beginning of the warm-up (as in 03CubicBuilding). The largest time constant is
7.9 d with controller and 17.6 d without (Assignment 3), and 20 °C is close to the concrete temperature in
December, so after 30 days the initial value hardly matters."""),
    code("""x0 = np.full(len(states), 20.0)"""),
    md("""## 4 Integration in time

$K_p$ of each zone is $10^5$ W/K (on) or 0 (off), which gives four models, each discretised once with implicit
Euler:

$$\\theta_{C,k+1} = \\Phi\\,\\theta_{C,k} + \\Gamma u_k, \\qquad \\Phi = (I - \\Delta t A_s)^{-1}, \\quad
\\Gamma = \\Phi\\,\\Delta t B_s$$

$q_{HVAC} = K_p (T_{sp} - \\theta_{air})$ is evaluated with the air temperature at the end of the step; the
controller decides on the measured air temperature at the start of the step.

`simulate` has two controllers (MPC in section 6). The P-controller of Assignments 1–3 is always on with the
heating set-point, so it also cools whenever the air is above it. The dead-band controller (as in tutorial
05SwitchModels) switches a zone on with $T_{sp} = T_h$ below $T_h$, with $T_{sp} = T_c$ above $T_c$, and off in
between."""),
    code("""K_on = 1e5                                                       # W/K, gain of the P-controller when on


def discrete(on1, on2, dt_=dt):
    \"\"\"Implicit Euler model with the controller of zone 1 / zone 2 on (1) or off (0).\"\"\"
    G_K = G.copy()
    G_K[k_ctrl[0], k_ctrl[0]] = K_on * on1
    G_K[k_ctrl[1], k_ctrl[1]] = K_on * on2
    As, Bs, Cs, Ds = dm4bem.tc2ss(A, G_K, C, b_ptr, f_ptr, y_ptr)
    Phi = np.linalg.inv(np.eye(len(As)) - dt_ * As)
    return {'Phi': Phi, 'Gam': Phi @ (dt_ * Bs), 'Cs': Cs}


models = {(i, j): discrete(i, j) for i in (0, 1) for j in (0, 1)}
Cs = models[(0, 0)]['Cs']                                        # same in all models: the air nodes are states


def simulate(control, U, x0, k_start=0, k_end=None, mods=models, sched=None):
    \"\"\"Simulation from step k_start to k_end. Returns y (air °C), q_HVAC (W) per zone and the states.

    sched = (T_h, T_c, occupied) on the time axis of U; default: the 15-min schedules.\"\"\"
    T_h, T_c, occupied = sched or (globals()['T_h'], globals()['T_c'], globals()['occupied'])
    Ua = U.to_numpy()
    k_end = k_end or len(Ua)
    m_ = k_end - k_start
    y, q, X = np.zeros((m_, 2)), np.zeros((m_, 2)), np.zeros((m_, len(x0)))
    x = x0.copy()
    for i, k in enumerate(range(k_start, k_end)):
        X[i] = x
        θ_k = Cs @ x                                             # air at the start of the step (measurement)
        if control == 'P':
            on, sp = (1, 1), np.array([T_h[k], T_h[k]])
        else:                                                    # dead-band
            on, sp = [0, 0], np.array([T_h[k], T_h[k]])
            for j in range(2):
                if θ_k[j] < T_h[k]:
                    on[j], sp[j] = 1, T_h[k]
                elif θ_k[j] > T_c[k]:
                    on[j], sp[j] = 1, T_c[k]
            on = tuple(on)
        m = mods[on]
        u = Ua[k].copy()
        u[j_sp] = sp
        x = m['Phi'] @ x + m['Gam'] @ u
        y[i] = Cs @ x                                            # air at the end of the step
        q[i] = K_on * np.array(on) * (sp - y[i])
    return pd.DataFrame({'θ1': y[:, 0], 'θ2': y[:, 1], 'q1': q[:, 0], 'q2': q[:, 1],
                         'T_h': T_h[k_start:k_end], 'T_c': T_c[k_start:k_end],
                         'occupied': occupied[k_start:k_end]}, index=U.index[k_start:k_end]), X


res, X = {}, {}
for c in ('P', 'dead-band'):
    r, X[c] = simulate(c, U, x0)
    res[c] = r.iloc[k0:]                                          # without the warm-up
    X[c] = X[c][k0:]
print('simulated: ' + ', '.join(res))"""),
    md("""## 5 Results"""),
    code("""S_tot = 2 * S_plan                                               # m², floor area of both zones


def key_figures(r):
    \"\"\"Annual energy, peak loads and comfort of a simulation result.\"\"\"
    q = r[['q1', 'q2']].to_numpy()
    θ = r[['θ1', 'θ2']].to_numpy()
    occ = r['occupied'].to_numpy()[:, None]
    under = np.clip(r[['T_h']].to_numpy() - 0.5 - θ, 0, None) * occ   # K below T_h - 0.5 K during occupancy
    over = np.clip(θ - r[['T_c']].to_numpy() - 0.5, 0, None) * occ     # K above T_c + 0.5 K
    hours = len(r) * dt / 3600
    return pd.Series({
        'heating, kWh': np.clip(q, 0, None).sum() * dt / 3.6e6,
        'cooling, kWh': -np.clip(q, None, 0).sum() * dt / 3.6e6,
        'heating, kWh/m²': np.clip(q, 0, None).sum() * dt / 3.6e6 / S_tot * 8760 / hours,
        'cooling, kWh/m²': -np.clip(q, None, 0).sum() * dt / 3.6e6 / S_tot * 8760 / hours,
        'peak heating (sum of zones), W/m²': q.max(axis=0).sum() / S_tot,
        'peak cooling (sum of zones), W/m²': -q.min(axis=0).sum() / S_tot,
        'under-heating, Kh': under.sum() * dt / 3600,
        'over-heating, Kh': over.sum() * dt / 3600})


fig_tab = pd.DataFrame({c: key_figures(r) for c, r in res.items()})
r = res['dead-band']
for j, name in ((1, 'zone 1 (south)'), (2, 'zone 2 (north)')):
    print(f'dead-band, {name}: heating {r[f"q{j}"].clip(lower=0).sum() * dt / 3.6e6 / S_plan:.1f} kWh/m², '
          f'cooling {-r[f"q{j}"].clip(upper=0).sum() * dt / 3.6e6 / S_plan:.1f} kWh/m²')
fig_tab.round(1)"""),
    md("""A winter and a summer week (Monday to Sunday), P-controller and dead-band. Grey band: between heating and
cooling set-point."""),
    code("""weeks = {'winter week': (f'{year}-01-16', f'{year}-01-22 23:45'),
         'summer week': (f'{year}-07-17', f'{year}-07-23 23:45')}


def plot_week(rs, title):
    \"\"\"Air temperatures and q_HVAC of several results in one week.\"\"\"
    fig, axs = plt.subplots(2, 2, figsize=(13, 6), sharex='col')
    for col, (wname, (a, b)) in enumerate(weeks.items()):
        for name, r in rs.items():
            r = r.loc[a:b]
            for j, ls in ((1, '-'), (2, '--')):
                axs[0, col].plot(r.index, r[f'θ{j}'], ls, lw=1, label=f'{name}, zone {j}')
                axs[1, col].plot(r.index, r[f'q{j}'] / 1000, ls, lw=1, label=f'{name}, zone {j}')
        r = next(iter(rs.values())).loc[a:b]
        axs[0, col].fill_between(r.index, r['T_h'], r['T_c'], color='grey', alpha=0.15, step='post')
        axs[0, col].plot(w.loc[a:b, 'T_o'], color='grey', lw=0.8, label='T_o')
        axs[0, col].set(title=f'{title}: {wname}', ylabel='θ, °C')
        axs[1, col].set(ylabel='q_HVAC, kW (> 0 heating)')
        axs[1, col].tick_params(axis='x', rotation=30)
    for ax in axs.flat:
        ax.grid(alpha=0.3)
    axs[0, 1].legend(fontsize=7, loc='upper right')
    axs[1, 1].legend(fontsize=7, loc='lower right')
    plt.tight_layout()
    plt.show()


plot_week(res, 'P-controller and dead-band')"""),
    code("""mon = pd.DataFrame({
    'heating P': res['P'][['q1', 'q2']].clip(lower=0).sum(axis=1),
    'cooling P': res['P'][['q1', 'q2']].clip(upper=0).sum(axis=1),
    'heating dead-band': res['dead-band'][['q1', 'q2']].clip(lower=0).sum(axis=1),
    'cooling dead-band': res['dead-band'][['q1', 'q2']].clip(upper=0).sum(axis=1),
}).resample('MS').sum() * dt / 3.6e6 / S_tot
fig, ax = plt.subplots(figsize=(10, 3.5))
x_m = np.arange(12)
for i, (col, color, hatch) in enumerate([('heating P', 'tab:red', '//'), ('cooling P', 'tab:blue', '//'),
                                         ('heating dead-band', 'tab:red', ''), ('cooling dead-band', 'tab:blue', '')]):
    ax.bar(x_m + (i // 2 - 0.5) * 0.4, mon[col], 0.4, color=color, alpha=0.5 if hatch else 0.9, hatch=hatch,
           label=col)
ax.set(xticks=x_m, xticklabels=mon.index.strftime('%b'), ylabel='kWh/m² per month',
       title='Monthly heating (> 0) and cooling (< 0) energy, both zones')
ax.axhline(0, color='k', lw=0.6)
ax.legend(fontsize=8, ncol=2)
ax.grid(axis='y', alpha=0.3)
plt.show()"""),
    md("""Time step check: both weeks again with the dead-band controller at 5 min, starting from the states of the
annual simulation."""),
    code("""dt_f = 300
models_f = {key: discrete(*key, dt_=dt_f) for key in models}
U_f = U.resample(f'{dt_f}s').ffill()                              # inputs constant within 15 min
t_f = U_f.index
sched_f = tuple(pd.Series(v, t).reindex(t_f, method='ffill').to_numpy() for v in (T_h, T_c, occupied))
for wname, (a, b) in weeks.items():
    r15 = res['dead-band'].loc[a:b]
    ka, kb = t_f.get_loc(pd.Timestamp(a)), t_f.get_loc(pd.Timestamp(b)) + 1
    xa = X['dead-band'][res['dead-band'].index.get_loc(pd.Timestamp(a))]
    r5, _ = simulate('dead-band', U_f, xa, ka, kb, models_f, sched_f)
    e15 = r15[['q1', 'q2']].abs().sum().sum() * dt / 3.6e6
    e5 = r5[['q1', 'q2']].abs().sum().sum() * dt_f / 3.6e6
    print(f'{wname}: heating + cooling {e15:.1f} kWh (Δt = 15 min), {e5:.1f} kWh (Δt = 5 min), '
          f'difference {(e15 - e5) / e5:+.1%}; max |Δθ_air| = '
          f'{np.abs(r15["θ1"].values - r5["θ1"].resample(f"{dt}s").last().values[:len(r15)]).max():.2f} K')"""),
    md("""## 6 Model predictive control (MPC)

The MPC uses the model with $K_p = 0$ and adds $q_{HVAC}$ of each zone to the air node. Every hour it solves a
linear program over 24 h with a perfect forecast of all inputs; $q_{HVAC}$ is decided per 15 min for the first 4 h
and hourly after that, the air temperature is checked every step:

$$\\min \\; \\sum_h \\left(q_{h} + q_{c}\\right) \\cdot 1\\,\\text{h} \\; + \\; \\lambda \\left(\\hat q_h + \\hat q_c\\right)
\\; + \\; \\rho \\sum s$$

$$T_h - s \\le \\theta_{air} \\le T_c + s, \\qquad 0 \\le q_h \\le \\hat q_h, \\quad 0 \\le q_c \\le \\hat q_c,
\\quad q_h, q_c \\le P_{max}, \\qquad s \\ge 0$$

with $q_{HVAC} = q_h - q_c$ and $\\hat q_h$, $\\hat q_c$ the peak powers per zone in the horizon. The air
temperature is $\\theta_{air} = \\theta_{free} + H q$, with $\\theta_{free}$ the prediction for $q = 0$ and $H$ from
$C_s \\Phi^m \\Gamma_q$. The slack $s$ (penalty $\\rho$ = 1000 kWh/K) keeps the problem feasible;
$P_{max}$ = 30 kW per zone (assumption, about 5 times the design heating load). The first hour is applied, then
the plan is recalculated (receding horizon).

"MPC, energy" uses $\\lambda = 0$; "MPC, energy + peak" uses $\\lambda$ = 4 kWh per kW (assumption, e.g. a demand
charge). Both run for the winter and summer week from the dead-band states."""),
    code("""m0 = models[(0, 0)]
Phi0, Gam0 = m0['Phi'], m0['Gam']
j_q = [jf[air1], jf[air2]]                                       # inputs of the air nodes: q_HVAC is added here
N, P_max, rho = 96, 30e3, 1000.0                                 # horizon: 96 steps of 15 min = 24 h
block = np.r_[np.arange(16), 16 + (np.arange(16, N) - 16) // 4]  # decision of each step: 15 min for 4 h, then hourly
n_h = block.max() + 1                                            # number of decisions per zone
hours = np.bincount(block) * dt / 3600                           # h, duration of each decision

Hm, Pm = [], np.eye(len(states))                                 # Hm[m] = Cs Φ^m Γ_q (2 × 2)
for m in range(N):
    Hm.append(Cs @ Pm @ Gam0[:, j_q])
    Pm = Phi0 @ Pm
H = np.zeros((2 * N, 2 * N))                                     # air at the end of steps k … k+N-1 <- q
for r_ in range(N):
    for c_ in range(r_ + 1):
        H[2 * r_:2 * r_ + 2, 2 * c_:2 * c_ + 2] = Hm[r_ - c_]
E_h = np.zeros((2 * N, 2 * n_h))                                 # q of the time steps <- q of the decisions
for i in range(N):
    E_h[2 * i:2 * i + 2, 2 * block[i]:2 * block[i] + 2] = np.eye(2)
H_h = H @ E_h
n_v = 6 * n_h + 4                                                # variables: q_h, q_c, s (per decision, zone), peaks
Z = np.zeros((2 * N, 4))
A_comfort = np.block([[-H_h, H_h, -E_h, Z], [H_h, -H_h, -E_h, Z]])
A_peak = np.zeros((4 * n_h, n_v))                                # q_h - peak_h ≤ 0, q_c - peak_c ≤ 0
for i in range(2 * n_h):
    A_peak[i, i], A_peak[i, 6 * n_h + i % 2] = 1, -1
    A_peak[2 * n_h + i, 2 * n_h + i], A_peak[2 * n_h + i, 6 * n_h + 2 + i % 2] = 1, -1
A_ub = np.vstack([A_comfort, A_peak])
bounds = [(0, P_max)] * (4 * n_h) + [(0, None)] * (2 * n_h + 4)


def mpc(U, x0, k_start, k_end, lam):
    \"\"\"MPC from step k_start to k_end with peak weight lam (kWh per kW). Returns the same table as simulate.\"\"\"
    e = np.repeat(hours, 2) * 1e-3                               # kWh per W of each decision
    c_lp = np.r_[e, e, np.full(2 * n_h, rho), np.full(4, lam * 1e-3)]
    Ua = U.to_numpy()
    y, q = np.zeros((k_end - k_start, 2)), np.zeros((k_end - k_start, 2))
    x = x0.copy()
    for i0 in range(0, k_end - k_start, 4):
        k = k_start + i0
        xf, y_free = x.copy(), np.zeros(2 * N)                   # free prediction (q = 0)
        for m in range(N):
            xf = Phi0 @ xf + Gam0 @ Ua[min(k + m, len(Ua) - 1)]
            y_free[2 * m:2 * m + 2] = Cs @ xf
        kk = np.minimum(np.arange(k, k + N), len(Ua) - 1)        # set-points of the steps k … k+N-1
        lo, hi = np.repeat(T_h[kk], 2), np.repeat(T_c[kk], 2)
        sol = linprog(c_lp, A_ub=A_ub, b_ub=np.r_[y_free - lo, hi - y_free, np.zeros(4 * n_h)],
                      bounds=bounds, method='highs')
        plan = (sol.x[:2 * n_h] - sol.x[2 * n_h:4 * n_h]).reshape(n_h, 2)
        for m in range(min(4, k_end - k)):                        # apply the first hour of the plan
            u = Ua[k + m].copy()
            u[j_q] += plan[m]
            x = Phi0 @ x + Gam0 @ u
            y[i0 + m], q[i0 + m] = Cs @ x, plan[m]               # air at the end of the step, q during the step
    sl = slice(k_start, k_end)
    return pd.DataFrame({'θ1': y[:, 0], 'θ2': y[:, 1], 'q1': q[:, 0], 'q2': q[:, 1], 'T_h': T_h[sl], 'T_c': T_c[sl],
                         'occupied': occupied[sl]}, index=U.index[sl])


res_w = {'dead-band': [], 'MPC, energy': [], 'MPC, energy + peak': []}
for wname, (a, b) in weeks.items():
    ka, kb = t.get_loc(pd.Timestamp(a)), t.get_loc(pd.Timestamp(b)) + 1
    xa = X['dead-band'][res['dead-band'].index.get_loc(pd.Timestamp(a))]
    res_w['dead-band'].append(res['dead-band'].loc[a:b])
    res_w['MPC, energy'].append(mpc(U, xa, ka, kb, lam=0))
    res_w['MPC, energy + peak'].append(mpc(U, xa, ka, kb, lam=4))
res_w = {c: pd.concat(v) for c, v in res_w.items()}
plot_week({c: res_w[c] for c in ('dead-band', 'MPC, energy + peak')}, 'dead-band and MPC (energy + peak)')"""),
    code("""tab_w = {}
for wname, (a, b) in weeks.items():
    for c, r in res_w.items():
        kf = key_figures(r.loc[a:b])
        tab_w[(wname, c)] = kf[['heating, kWh', 'cooling, kWh', 'peak heating (sum of zones), W/m²', 'peak cooling (sum of zones), W/m²',
                                'under-heating, Kh', 'over-heating, Kh']]
pd.DataFrame(tab_w).round(1)"""),
    md("""## 7 Discussion

Inputs: 1138 kWh/(m²·a) south, 732 east, 829 west, 468 north, 1449 roof (the time correction mostly moves
radiation between east and west); internal gains 58.7 kWh/(m²·a).

P-controller: one set-point for heating and cooling, so it cools to 21 °C by day and 18 °C at night in summer and
cools the sunlit south zone in winter. 11.8 kWh/(m²·a) heating and 65.7 kWh/(m²·a) cooling: fine for steady state
and step response, not for a year.

Dead-band: 6.9 kWh/(m²·a) heating, 23.9 kWh/(m²·a) cooling (south 3.9 / 25.0, north 9.9 / 22.8), against 8.1 and
12.4 kWh/(m²·a) from SIA 2024 (room type 3.01, Chur). Cooling is about double: no heat-recovery bypass and no night
ventilation (26–27 °C inside at night, often 10–20 °C outside), blinds only above 200 W/m² so the north windows
(60 % of the façade) are hardly shaded, convective gains and no pre-cooling before 07:00. The peak heating load of
38.6 W/m² exceeds the design load of 24.6 W/m² (Assignment 2) since the set-point jumps from 18 to 21 °C at 07:00
and the unlimited controller re-heats in about one step. In winter the south zone floats up to about 24 °C while
the north zone is heated, partly via the interior wall; in summer the unshaded north zone needs more cooling. The
symmetric P-controller of the active mode also gives 19 kWh cooling and 10 kWh heating per year in the wrong mode.

Time step: 5 min instead of 15 min changes the energy by less than 1 % and the air temperature by at most 0.4 K.

MPC: with energy only it matches the dead-band controller within 1 %, as both heat as late as possible. With the
peak in the cost (4 kWh per kW) it pre-heats into the thermal mass and cuts the winter-week peak by 34 %
(27.1 to 17.9 W/m²) for about 6 % more heating energy. The peak term only covers the 24 h horizon; a monthly
demand charge would need the peak already reached as a constraint. The summer peak stays, because pre-cooling the
concrete costs more than it saves; night ventilation (not modelled) would help more.

Limits: ideal controllers (dead-band without power limit, MPC with 30 kW and perfect forecast), sensible heat only,
convective gains, no night ventilation or bypass, one typical year (weather about half an hour early, as values sit
at the start of their hour and are interpolated), hypotheses of Assignment 1 (e.g. star network for long-wave
radiation, constant coefficients)."""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python"}
nbf.write(nb, sys.argv[1])
print("written", sys.argv[1])
