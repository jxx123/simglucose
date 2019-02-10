# simglucose
[![Downloads](https://pepy.tech/badge/simglucose)](https://pepy.tech/project/simglucose)
[![Downloads](https://pepy.tech/badge/simglucose/month)](https://pepy.tech/project/simglucose)
[![Downloads](https://pepy.tech/badge/simglucose/week)](https://pepy.tech/project/simglucose)

A Type-1 Diabetes simulator implemented in Python for Reinforcement Learning purpose

This simulator is a python implementation of the FDA-approved [UVa/Padova Simulator (2008 version)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4454102/) for research purpose only. The simulator includes 30 virtual patients, 10 adolescents, 10 adults, 10 children. 
 
 **HOW TO CITE**: Jinyu Xie. Simglucose v0.2.1 (2018) \[Online\]. Avaible: https://github.com/jxx123/simglucose. Accessed on: Month-Date-Year.

 - Note: simglucose only supports python3.


| Animation                                                                                         | CVGA Plot                                                                      | BG Trace Plot                                                                                    | Risk Index Stats                                                                                                 |
|---------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| ![animation screenshot](https://github.com/jxx123/simglucose/blob/master/screenshots/animate.png) | ![CVGA](https://github.com/jxx123/simglucose/blob/master/screenshots/CVGA.png) | ![BG Trace Plot](https://github.com/jxx123/simglucose/blob/master/screenshots/BG_trace_plot.png) | ![Risk Index Stats](https://github.com/jxx123/simglucose/blob/master/screenshots/risk_index.png) |

  <!-- ![Zone Stats](https://github.com/jxx123/simglucose/blob/master/screenshots/zone_stats.png) -->

## Main Features
- Simulation enviroment follows [OpenAI gym](https://github.com/openai/gym) and [rllab](https://github.com/rll/rllab) APIs. It returns observation, reward, done, info at each step, which means the simulator is "reinforcement-learning-ready".
- Supports customized reward function. The reward function is a function of blood glucose measurements in the last hour. By default, the reward at each step is `risk[t-1] - risk[t]`. `risk[t]` is the risk index at time `t` defined in this [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2903980/pdf/dia.2008.0138.pdf). 
- Supports parallel computing. The simulator simulates mutliple patients parallelly using [pathos multiprocessing package](https://github.com/uqfoundation/pathos) (you are free to turn parallel off by setting `parallel=False`).
- The simulator provides a random scenario generator (`from simglucose.simulation.scenario_gen import RandomScenario`) and a customized scenario generator (`from simglucose.simulation.scenario import CustomScenario`). Commandline user-interface will guide you through the scenario settings.
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
### Use simglucose as a simulator and test controllers
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
        self.init_state = init_state
        self.state = init_state

    def policy(self, observation, reward, done, **info):
        '''
        Every controller must have this implementation!
        ----
        Inputs:
        observation - a namedtuple defined in simglucose.simulation.env. For
                      now, it only has one entry: blood glucose level measured
                      by CGM sensor.
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

    def reset(self):
        '''
        Reset the controller state to inital state, must be implemented
        '''
        self.state = self.init_state


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
### OpenAI Gym usage
- Using default reward
```python
import gym

# Register gym environment. By specifying kwargs,
# you are able to choose which patient to simulate.
# patient_name must be 'adolescent#001' to 'adolescent#010',
# or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
from gym.envs.registration import register
register(
    id='simglucose-adolescent2-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adolescent#002'}
)

env = gym.make('simglucose-adolescent2-v0')

observation = env.reset()
for t in range(100):
    env.render(mode='human')
    print(observation)
    # Action in the gym environment is a scalar
    # representing the basal insulin, which differs from
    # the regular controller action outside the gym
    # environment (a tuple (basal, bolus)).
    # In the perfect situation, the agent should be able
    # to control the glucose only through basal instead
    # of asking patient to take bolus
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t + 1))
        break
```
- Customized reward function
```python
import gym
from gym.envs.registration import register


def custom_reward(BG_last_hour):
    if BG_last_hour[-1] > 180:
        return -1
    elif BG_last_hour[-1] < 70:
        return -2
    else:
        return 1


register(
    id='simglucose-adolescent2-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adolescent#002',
            'reward_fun': custom_reward}
)

env = gym.make('simglucose-adolescent2-v0')

reward = 1
done = False

observation = env.reset()
for t in range(200):
    env.render(mode='human')
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(observation)
    print("Reward = {}".format(reward))
    if done:
        print("Episode finished after {} timesteps".format(t + 1))
        break
```

### rllab usage
```python
from rllab.algos.ddpg import DDPG
from rllab.envs.normalized_env import normalize
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from rllab.envs.gym_env import GymEnv
from gym.envs.registration import register

register(
    id='simglucose-adolescent2-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adolescent#002'}
)

env = GymEnv('simglucose-adolescent2-v0')
env = normalize(env)

policy = DeterministicMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
)

es = OUStrategy(env_spec=env.spec)

qf = ContinuousMLPQFunction(env_spec=env.spec)

algo = DDPG(
    env=env,
    policy=policy,
    es=es,
    qf=qf,
    batch_size=32,
    max_path_length=100,
    epoch_length=1000,
    min_pool_size=10000,
    n_epochs=1000,
    discount=0.99,
    scale_reward=0.01,
    qf_learning_rate=1e-3,
    policy_learning_rate=1e-4
)
algo.train()
```

## Advanced Usage
You can create the simulation objects, and run batch simulation. For example,
```python
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
from datetime import timedelta
from datetime import datetime

# specify start_time as the beginning of today
now = datetime.now()
start_time = datetime.combine(now.date(), datetime.min.time())

# --------- Create Random Scenario --------------
# Specify results saving path
path = './results'

# Create a simulation environment
patient = T1DPatient.withName('adolescent#001')
sensor = CGMSensor.withName('Dexcom', seed=1)
pump = InsulinPump.withName('Insulet')
scenario = RandomScenario(start_time=start_time, seed=1)
env = T1DSimEnv(patient, sensor, pump, scenario)

# Create a controller
controller = BBController()

# Put them together to create a simulation object
s1 = SimObj(env, controller, timedelta(days=1), animate=False, path=path)
results1 = sim(s1)
print(results1)

# --------- Create Custom Scenario --------------
# Create a simulation environment
patient = T1DPatient.withName('adolescent#001')
sensor = CGMSensor.withName('Dexcom', seed=1)
pump = InsulinPump.withName('Insulet')
# custom scenario is a list of tuples (time, meal_size)
scen = [(7, 45), (12, 70), (16, 15), (18, 80), (23, 10)]
scenario = CustomScenario(start_time=start_time, scenario=scen)
env = T1DSimEnv(patient, sensor, pump, scenario)

# Create a controller
controller = BBController()

# Put them together to create a simulation object
s2 = SimObj(env, controller, timedelta(days=1), animate=False, path=path)
results2 = sim(s2)
print(results2)


# --------- batch simulation --------------
# Re-initialize simulation objects
s1.reset()
s2.reset()

# create a list of SimObj, and call batch_sim
s = [s1, s2]
results = batch_sim(s, parallel=True)
print(results)
```

Run analysis offline
```python
from simglucose.analysis.report import report
import pandas as pd
import os
import glob

# the path where results are saved
path = os.path.join('.', 'results', '2017-12-31_17-46-32')
os.chdir(path)
# find all csv with pattern *#*.csv, e.g. adolescent#001.csv
filename = glob.glob('*#*.csv')
name = [_f[:-4] for _f in filename]   # get the filename without extension
df = pd.concat([pd.read_csv(f, index_col=0) for f in filename], keys=name)
report(df)
```
## Release Notes
### 9/10/2018
- Controller `policy` method gets access to all the current patient state through `info['patient_state']`.
### 2/26/2018
- Support customized reward function.
### 1/10/2018
- Added workaround to select patient when make gym environment: register gym environment by passing kwargs of patient_name.
### 1/7/2018
- Added OpenAI gym support, use `gym.make('simglucose-v0')` to make the enviroment.
- Noticed issue: the patient name selection is not available in gym.make for now. The patient name has to be hard-coded in the constructor of `simglucose.envs.T1DSimEnv`.

## Reporting issues
Shoot me any bugs, enhancements or even discussion by [creating issues](https://github.com/jxx123/simglucose/issues/new).

## How to contribute
The following instruction is originally from the [contribution instructions of sklearn](https://github.com/scikit-learn/scikit-learn/blob/master/CONTRIBUTING.md).

The preferred workflow for contributing to simglucose is to fork the
[main repository](https://github.com/jxx123/simglucose) on
GitHub, clone, and develop on a branch. Steps:

1. Fork the [project repository](https://github.com/jxx123/simglucose)
   by clicking on the 'Fork' button near the top right of the page. This creates
   a copy of the code under your GitHub user account. For more details on
   how to fork a repository see [this guide](https://help.github.com/articles/fork-a-repo/).

2. Clone your fork of the simglucose repo from your GitHub account to your local disk:

   ```bash
   $ git clone git@github.com:YourLogin/simglucose.git
   $ cd simglucose
   ```

3. Create a ``feature`` branch to hold your development changes:

   ```bash
   $ git checkout -b my-feature
   ```

   Always use a ``feature`` branch. It's good practice to **never work on the ``master`` branch**!

4. Develop the feature on your feature branch. Add changed files using ``git add`` and then ``git commit`` files:

   ```bash
   $ git add modified_files
   $ git commit
   ```

   to record your changes in Git, then push the changes to your GitHub account with:

   ```bash
   $ git push -u origin my-feature
   ```

5. Follow [these instructions](https://help.github.com/articles/creating-a-pull-request-from-a-fork)
to create a pull request from your fork. This will send an email to the committers.

(If any of the above seems like magic to you, please look up the
[Git documentation](https://git-scm.com/documentation) on the web, or ask a friend or another contributor for help.)
