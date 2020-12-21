import gym
import unittest
from simglucose.controller.basal_bolus_ctrller import BBController
from datetime import datetime
from gym.envs.registration import register


class TestSeed(unittest.TestCase):
    def test_changing_seed_generates_different_results(self):
        register(
            id='simglucose-adult1-v0',
            entry_point='simglucose.envs:T1DSimEnv',
            kwargs={'patient_name': 'adult#001'}
        )

        env = gym.make('simglucose-adult1-v0')

        env.seed(0)
        observation_seed0 = env.reset()
        self.assertEqual(env.env.scenario.start_time, datetime(2018, 1, 1, 16, 0, 0))

        env.seed(1000)
        observation_seed1 = env.reset()
        self.assertEqual(env.env.scenario.start_time, datetime(2018, 1, 1, 10, 0, 0))

        self.assertNotEqual(observation_seed0, observation_seed1)


if __name__ == '__main__':
    unittest.main()
