import gym
import unittest
from gym.envs.registration import register

register(
        id='simglucose-adult2-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adult#002'}
        )


class TestReset(unittest.TestCase):
    def test_reset_changes_observation_when_seed_is_fixed(self):
        env = gym.make('simglucose-adult2-v0')

        env.seed(0)
        observation0 = env.reset()
        start_time0 = env.env.scenario.start_time
        scenario0 = env.env.scenario.scenario

        observation1 = env.reset()
        start_time1 = env.env.scenario.start_time
        scenario1 = env.env.scenario.scenario

        self.assertNotEqual(observation0, observation1)
        self.assertNotEqual(start_time0, start_time1)
        self.assertNotEqual(scenario0, scenario1)

    def test_reset_change_is_deterministic_when_seed_is_fixed(self):
        env = gym.make('simglucose-adult2-v0')

        env.seed(0)
        observation0 = env.reset()
        start_time0 = env.env.scenario.start_time
        scenario0 = env.env.scenario.scenario

        observation1 = env.reset()
        start_time1 = env.env.scenario.start_time
        scenario1 = env.env.scenario.scenario
        
        env.seed(0)
        observation2 = env.reset()
        start_time2 = env.env.scenario.start_time
        scenario2 = env.env.scenario.scenario

        observation3 = env.reset()
        start_time3 = env.env.scenario.start_time
        scenario3 = env.env.scenario.scenario

        self.assertEqual(observation0, observation2)
        self.assertEqual(observation1, observation3)

        self.assertEqual(start_time0, start_time2)
        self.assertEqual(start_time1, start_time3)

        self.assertEqual(scenario0, scenario2)
        self.assertEqual(scenario1, scenario3)

    def test_reset_change_is_random_when_seed_is_different(self):
        env = gym.make('simglucose-adult2-v0')

        env.seed(0)
        observation0 = env.reset()
        start_time0 = env.env.scenario.start_time
        scenario0 = env.env.scenario.scenario

        observation1 = env.reset()
        start_time1 = env.env.scenario.start_time
        scenario1 = env.env.scenario.scenario
        
        env.seed(1)
        observation2 = env.reset()
        start_time2 = env.env.scenario.start_time
        scenario2 = env.env.scenario.scenario

        observation3 = env.reset()
        start_time3 = env.env.scenario.start_time
        scenario3 = env.env.scenario.scenario

        self.assertNotEqual(observation0, observation2)
        self.assertNotEqual(observation1, observation3)

        self.assertNotEqual(start_time0, start_time2)
        self.assertNotEqual(start_time1, start_time3)

        self.assertNotEqual(scenario0, scenario2)
        self.assertNotEqual(scenario1, scenario3)

if __name__ == '__main__':
    unittest.main()
