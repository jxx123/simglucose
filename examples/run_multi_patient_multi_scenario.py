from datetime import datetime
import gym
import simglucose
from simglucose.simulation.scenario import CustomScenario
import numpy as np


start_time = datetime(2018, 1, 1, 0, 0, 0)
meal_scenario_1 = CustomScenario(start_time=start_time, scenario=[(1, 20)])
meal_scenario_2 = CustomScenario(start_time=start_time, scenario=[(3, 15)])


patient_name = [
    "adult#001",
    "adult#002",
    "adult#003",
    "adult#004",
    "adult#005",
    "adult#006",
    "adult#007",
    "adult#008",
    "adult#009",
    "adult#010",
]

gym.envs.register(
    id="env-v0",
    entry_point="simglucose.envs:T1DSimEnv",
    kwargs={
        "patient_name": patient_name,
        "custom_scenario": [meal_scenario_1, meal_scenario_2],
    },
)

env = gym.make("env-v0")

env.reset()

min_insulin = env.action_space.low
max_insulin = env.action_space.high

observation = env.reset()
for t in range(100):
    env.render(mode="human")
    # action = np.random.uniform(min_insulin, max_insulin)

    action = observation.CGM * 0.0005
    if observation.CGM < 120:
        action = 0

    # print(action)
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t + 1))
        break
