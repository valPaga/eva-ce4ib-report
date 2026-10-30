# Single-storey two-zone office building in Chur – dynamic thermal model

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/valPaga/eva-ce4ib-report/HEAD?labpath=index.ipynb)

**Team:** Valerio Paganini

Project report of the course *Dynamic Models for Building Energy Management*
([dm4bem](https://github.com/cghiaus/dm4bem), Ch. Ghiaus), HS26.

**Run online:** <https://mybinder.org/v2/gh/valPaga/eva-ce4ib-report/HEAD?labpath=index.ipynb>
(the first start takes a few minutes while MyBinder builds the environment).

## Report

Start with [`index.ipynb`](index.ipynb), then the assignments in order:

| Notebook | Content |
|---|---|
| [Assignment1_Model.ipynb](Assignment1_Model.ipynb) | building, hypotheses, thermal circuit, matrices $A$, $G$, $C$, vectors $b$, $f$, $y$ |
| [Assignment2_SteadyState.ipynb](Assignment2_SteadyState.ipynb) | DAE and state-space representation, steady-state results |
| [Assignment3_StepResponse.ipynb](Assignment3_StepResponse.ipynb) | time constants, time step, step response |
| [Assignment4_Weather.ipynb](Assignment4_Weather.ipynb) | response to weather, P-controller, dead-band, model predictive control |

## Files

- `environment.yml` – environment for MyBinder (Python 3.12, numpy 1.26, pandas 2.2, matplotlib 3.9, scipy 1.13);
  `requirements.txt` – the same versions for a local `pip` installation.
- `dm4bem.py` – course library, copy of [cghiaus/dm4bem](https://github.com/cghiaus/dm4bem) (MIT licence).
- `daten/` – weather data Chur, TMYx 2004–2018, from [climate.onebuilding.org](https://climate.onebuilding.org).
- `figures/` – thermal circuit of Assignment 1.
- `tools/` – scripts that generate the notebooks.

## Run locally

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt jupyterlab
.venv/Scripts/jupyter lab index.ipynb
```
