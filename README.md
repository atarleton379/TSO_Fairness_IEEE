# Group 11: Assignment 1 of Course 46755 Renewables in electricity markets
## Overview
The task of the assignment was to develop a variety of market-clearing optimization models. In total, the assignment consists of six tasks, with model names corresponding to each task. Since task four does not require an optimization model, there is no `model4.py` in this repository. Each `model{X}.py` contains a class named `Model{X}`, composed of the following functions:

1. An `__init__` function that loads the required input data (via an `InputHandler` class from `input.py`) and runs the other class functions.
2. A `_run_model` function that builds and solves the optimization model.
3. One or more functions that analyze results and generate plots.

Use `main.py` to load the input data and pass it into the desired optimization model.

---
## Models
The following models were implemented:
- **Model 1** — Single-hour copper-plate day-ahead market clearing
- **Model 2** — Multiple hours copper-plate day-ahead market clearing with battery storage
- **Model 3 - Nodal** — Single-hour day-ahead market clearing incl. network constraints
- **Model 3 - Zonal** — Single-hour day-ahead market clearing incl. network constraints. Nodes grouped into zones.
- **Model 5** — Single-hour day-ahead market clearing followed by single-hour balancing market clearing.
- **Model 6** — Single-hour reserve capacity market clearing followed by single-hour day-ahead market clearing.

---


## Project Structure
```
RenewablesInElectricityMarkets/
├── code/
│   ├── input.py
│   ├── model1.py
│   ├── model2.py
│   ├── model3_nodal.py
│   ├── model3_zonal.py
│   ├── model5.py
│   ├── model6.py
│   ├── results_saver.py
├── results/
│   ├── sensitivity_results/
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation
```bash
pip install -r requirements.txt
```

---


## Nomenclature

| Symbol               | Description                                          | Unit     |
|----------------------|------------------------------------------------------|----------|
| **Demand**           |                                                      |          |
| $C^d_i$              | Bidding price of demand $i$                          | \$/MWh   |
| $d_i$                | Served demand $i$                                    | MWh      |
| $D$                  | Set of all demand nodes                              | -        |
| $D^{sum}_t$          | Total system load at hour $t$                        | MWh      |
| $\delta_i$           | Load distribution factor of demand $i$               | p.u.     |
| **Conventional Generation** |                                               |          |
| $C^{gen}_i$          | Bidding price of generator $i$                       | \$/MWh   |
| $p^{gen}_i$          | Power output of generator $i$                        | MWh      |
| $P^{gen,max}_i$      | Installed capacity of generator $i$                  | MW       |
| $G$                  | Set of all conventional generators                   | -        |
| **Wind Generation**  |                                                      |          |
| $p^{wind}_i$         | Power output of wind farm $i$                        | MWh      |
| $P^{wind,max}_i$     | Installed capacity of wind farm $i$                  | MW       |
| $CF^{wind}_{i,t}$    | Capacity factor of wind farm $i$ at hour $t$         | p.u.     |
| $W$                  | Set of all wind farms                                | -        |
| **Battery Storage**  |                                                      |          |
| $p^{ch}_{bat}$       | Charging power of battery                            | MW       |
| $p^{dis}_{bat}$      | Discharging power of battery                         | MW       |
| $P^{ch,max}_{bat}$   | Maximum charging power of battery                    | MW       |
| $P^{dis,max}_{bat}$  | Maximum discharging power of battery                 | MW       |
| $e_{bat}$            | Stored energy in battery at end of period            | MWh      |
| $E^{cap}_{bat}$      | Energy capacity of battery                           | MWh      |
| $\eta^{ch}$          | Charging efficiency of battery                       | p.u.     |
| $\eta^{dis}$         | Discharging efficiency of battery                    | p.u.     |
| **Network**          |                                                      |          |
| $p^{line}_i$         | Power flow on transmission line $i$                  | MW       |
| $\theta_n$           | Voltage angle at node $n$                            | rad      |
| $I$                  | Set of all transmission lines                        | -        |
| $p^{zone}_{(a,b)}$   | Net power flow between zone $a$ and zone $b$         | MW       |
| **Balancing Market** |                                                      |          |
| $b^{up}_j$           | Upward activation by BSP $j$                         | MWh      |
| $b^{down}_j$         | Downward activation by BSP $j$                       | MWh      |
| $b^{curt}_i$         | Involuntary demand curtailment of load $i$           | MWh      |
| $C^{+}_j$            | Upward regulation offer price of BSP $j$             | \$/MWh   |
| $C^{-}_j$            | Downward regulation offer price of BSP $j$           | \$/MWh   |
| $C^{curt}$           | Cost of involuntary demand curtailment               | \$/MWh   |
| $r^{up}_j$           | Upward reserve capacity offered by BSP $j$           | MW       |
| $r^{down}_j$         | Downward reserve capacity offered by BSP $j$         | MW       |
| $C^{res,up}_j$       | Upward reserve capacity cost of BSP $j$              | \$/MW    |
| $C^{res,down}_j$     | Downward reserve capacity cost of BSP $j$            | \$/MW    |