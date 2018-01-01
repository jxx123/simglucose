# simglucose
A Type-1 Diabetes simulator implemented in Python for Reinforcement Learning purpose

This simulator is a python implementation of [UVa/Padova Simulator (2008 version)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4454102/) for research purpose only. The simulator includes 30 virtual patients, 10 adolescents, 10 adults, 10 children. 

 - Note: simglucose only supports python3.

| Animation                                                                                         | CVGA Plot                                                                      | BG Trace Plot                                                                                    | Risk Index Stats                                                                                                 |
|---------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| ![animation screenshot](https://github.com/jxx123/simglucose/blob/master/screenshots/animate.png) | ![CVGA](https://github.com/jxx123/simglucose/blob/master/screenshots/CVGA.png) | ![BG Trace Plot](https://github.com/jxx123/simglucose/blob/master/screenshots/BG_trace_plot.png) | ![Risk Index Stats](https://github.com/jxx123/simglucose/blob/master/screenshots/risk_index.png) |

  <!-- ![Zone Stats](https://github.com/jxx123/simglucose/blob/master/screenshots/zone_stats.png) -->
## Release Notes, 12/31/2017
- Simulation enviroment follows [OpenAI gym](https://github.com/openai/gym) and [rllab](https://github.com/rll/rllab) APIs. It returns observation, reward, done, info at each step, which means the simulator is "reinforcement-learning-ready".
- The reward at each step is `10 - risk_index`. Customized reward is not supported for now. `risk_index` is defined in this [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2903980/pdf/dia.2008.0138.pdf). 
- Supports parallel computing. The simulator simulates mutliple patients parallelly using [pathos multiprocessing package](https://github.com/uqfoundation/pathos) (you are free to turn parallel off by setting `parallel=False`).
- The simulator provides a random scenario generator (`from simglucose.simulation.scenario_gen import RandomScenario`) and a customized scenario generator (`from simglucose.simulation.scenario import CustomScnenario`). Commandline user-interface will guide you through the scenario settings.
- The simulator provides the most basic basal-bolus controller for now. It provides very simple syntax to implement your own controller, like Model Predictive Control, PID control, reinforcement learning control, etc. 
- You can specify random seed in case you want to repeat your experiments.
- The simulator will generate several plots for performance analysis after simulation. The plots include blood glucose trace plot, Control Variability Grid Analysis (CVGA) plot, statistics plot of blood glucose in different zones, risk indices statistics plot.
- NOTE: `animate` and `parallel` cannot be set to `True` at the same time in macOS. Most backends of matplotlib in macOS is not thread-safe. Windows has not been tested. Let me know the results if anybody has tested it out.

## Installation
It is highly recommended to use `pip` to install `simglucose`, follow this [link](https://pip.pypa.io/en/stable/installing/) to install pip.

Auto installation:
```bash
pip install simglucose
```

Manual installation: 
```bash
git clone https://github.com/jxx123/simglucose.git
cd simglucose
```
If you have `pip` installed, then
```bash
pip install -e .
```
If you do not have `pip`, then
```bash
python setup.py install
```

If [rllab (optional)](https://github.com/rll/rllab) is installed, the package will utilize some functionalities in rllab.

Note: there might be some minor differences between auto install version and manual install version. Use `git clone` and manual installation to get the latest version.

## Quick Start
Run the simulator user interface
```python
from simglucose.simulation.user_interface import simulate
simulate()
```

You are free to implement your own controller, and test it in the simulator. For example,
```python
from simglucose.simulation.user_interface import simulate
from simglucose.controller.base import Controller, Action


class MyController(Controller):
    def __init__(self, init_state):
        self.state = init_state

    def policy(self, observation, reward, done, **info):
        '''
        Every controller must have this implementation!
        ----
        Inputs:
        observation - a namedtuple defined in simglucose.simulation.env. It has
                      CHO and CGM two entries.
        reward      - current reward returned by environment
        done        - True, game over. False, game continues
        info        - additional information as key word arguments,
                      simglucose.simulation.env.T1DSimEnv returns patient_name
                      and sample_time
        ----
        Output:
        action - a namedtuple defined at the beginning of this file. The
                 controller action contains two entries: basal, bolus
        '''
        self.state = observation
        action = Action(basal=0, bolus=0)
        return action


ctrller = MyController(0)
simulate(controller=ctrller)
```
These two examples can also be found in examples\ folder.

In fact, you can specify a lot more simulation parameters through `simulation`:
```python
simulate(sim_time=my_sim_time,
         scenario=my_scenario,
         controller=my_controller,
         start_time=my_start_time,
         save_path=my_save_path,
         animate=False,
         parallel=True)
```
