"""Build uebung/index.ipynb: start page of the report (Assignment 5) with links to Assignments 1–4."""
import sys

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# Single-storey two-zone office building in Chur – thermal model and simulation

**Author:** Valerio Paganini
**Date:** 30 October 2026

Project of the course *Dynamic Models for Building Energy Management* (dm4bem, Ch. Ghiaus), HS26.
This notebook is the start page of the report (Assignment 5). The report consists of four Jupyter notebooks;
Assignments 2, 3 and 4 run the earlier notebooks they need, so every result can be reproduced from the data in this
folder.

**GitHub:** <https://github.com/valPaga/eva-ce4ib-report>
**MyBinder:** [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/valPaga/eva-ce4ib-report/HEAD?labpath=index.ipynb)

## Contents

1. [Building](#Building)
2. [Report](#Report): Assignments 1–4
   - description of the building, hypotheses and thermal model → Assignment 1
   - mathematical model (DAE and state space) and implementation in Python → Assignments 1 and 2
   - steady-state results → Assignment 2
   - dynamic simulation results → Assignments 3 and 4
   - optimization of the HVAC control (dead-band, model predictive control) → Assignment 4
3. [Reproducibility](#Reproducibility)

## Building

Single-storey office building (SIA 2024 room type 3.01) above an unheated basement, plan 30 × 15 m. Zone 1 is the
south half, zone 2 the north half (225 m² each), separated by a concrete interior wall with a closed door. External
walls with 0.20 m mineral wool and 0.20 m concrete, roof with 0.16 m PUR and 0.28 m concrete, floor above the
basement with 0.14 m mineral wool and 0.30 m concrete; triple glazing on 60 % of every façade, external blinds.
Weather: Chur, typical year TMYx 2004–2018.

## Report

| Notebook | Content | Main results |
|---|---|---|
| [Assignment 1 – Model](Assignment1_Model.ipynb) | building, hypotheses, thermal circuit (155 nodes, 191 branches), matrices $A$, $G$, $C$, vectors $b$, $f$, $y$ | figure with the whole circuit and its numbering |
| [Assignment 2 – Steady state](Assignment2_SteadyState.ipynb) | DAE and state-space representation, three cases, comparison, visual check of the matrices | design heating load 5.5 kW per zone (24.6 W/m²) at −7.2 °C; DAE and state space agree to $10^{-12}$ °C |
| [Assignment 3 – Step response](Assignment3_StepResponse.ipynb) | time constants, time step, settling time, explicit and implicit Euler | time constants 8 s to 17.6 d; $\\Delta t_{max}$ = 16 s with the P-controller; settling time 31 to 70 d |
| [Assignment 4 – Response to weather](Assignment4_Weather.ipynb) | one year with weather, internal gains and blinds; P-controller, dead-band controller, model predictive control | dead-band: heating 6.9, cooling 23.9 kWh/(m²·a); MPC with peak cost: −34 % peak heating load in a winter week |

## Reproducibility

- **Run:** open the notebooks in the order of the table; Assignments 2 and 4 run Assignment 1 and 2 with `%run`,
  Assignment 3 runs Assignment 2. Running all four takes about 2–3 minutes, most of it Assignment 4.
- **Environment:** Python 3.12 with the versions in `requirements.txt` (the course library `dm4bem.py` does not
  run with numpy 2 or pandas 3). MyBinder uses `environment.yml`; `requirements.txt` holds the same versions for
  a local `pip` installation.
- **Data:** `daten/CHE_GR_Chur.067860_TMYx.2004-2018.epw` from [climate.onebuilding.org](https://climate.onebuilding.org).
- **Course library:** `dm4bem.py` (copy of [cghiaus/dm4bem](https://github.com/cghiaus/dm4bem), MIT licence).
- **Generators:** the notebooks are written by the scripts in `tools/` (`make_assignment1_notebook.py` …); the
  figure of Assignment 1 is in `figures/`.

The cell below checks the environment."""),
    code("""import sys
import numpy as np
import pandas as pd
import matplotlib
import scipy

print(f'Python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, '
      f'matplotlib {matplotlib.__version__}, scipy {scipy.__version__}')"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python"}
nbf.write(nb, sys.argv[1])
print("written", sys.argv[1])
