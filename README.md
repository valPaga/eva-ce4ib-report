# Two-zone office building in Chur – dynamic thermal model

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/valPaga/eva-ce4ib-report/HEAD?labpath=index.ipynb)

**Team:** Valerio Paganini

Report for the course *Dynamic Models for Building Energy Management* ([dm4bem](https://github.com/cghiaus/dm4bem),
Ch. Ghiaus), HS26. Start with [`index.ipynb`](index.ipynb), then:

1. [Assignment1_Model.ipynb](Assignment1_Model.ipynb) – building, thermal circuit, matrices
2. [Assignment2_SteadyState.ipynb](Assignment2_SteadyState.ipynb) – DAE, state space, steady state
3. [Assignment3_StepResponse.ipynb](Assignment3_StepResponse.ipynb) – time constants, time step, step response
4. [Assignment4_Weather.ipynb](Assignment4_Weather.ipynb) – one year with weather, dead-band, night ventilation, humid air and MPC

Environment in `environment.yml` (MyBinder) and `requirements.txt` (pip). `dm4bem.py` is a copy of the course
library (MIT), the weather file in `daten/` is from [climate.onebuilding.org](https://climate.onebuilding.org).
