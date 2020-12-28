import gym
import unittest
from datetime import datetime
from gym.envs.registration import register

register(
    id='simglucose-adult1-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adult#001'}
)


class TestSeed(unittest.TestCase):
    def test_changing_seed_generates_different_results(self):
        env = gym.make('simglucose-adult1-v0')

        env.seed(0)
        observation_seed0 = env.reset()
        self.assertEqual(env.env.scenario.start_time, datetime(2018, 1, 1, 23, 0, 0))

        env.seed(1000)
        observation_seed1 = env.reset()
        self.assertEqual(env.env.scenario.start_time, datetime(2018, 1, 1, 14, 0, 0))

        self.assertNotEqual(observation_seed0, observation_seed1)


if __name__ == '__main__':
    unittest.main()
