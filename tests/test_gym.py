import gym
import unittest
from simglucose.controller.basal_bolus_ctrller import BBController


class TestGym(unittest.TestCase):
    def test_gym_random_agent(self):
        from gym.envs.registration import register
        register(
            id='simglucose-adolescent2-v0',
            entry_point='simglucose.envs:T1DSimEnv',
            kwargs={'patient_name': 'adolescent#002'}
        )

        env = gym.make('simglucose-adolescent2-v0')
        ctrller = BBController()

        reward = 0
        done = False
        info = {'sample_time': 3,
                'patient_name': 'adolescent#002',
                'meal': 0}

        observation = env.reset()
        for t in range(200):
            env.render(mode='human')
            print(observation)
            # action = env.action_space.sample()
            ctrl_action = ctrller.policy(observation, reward, done, **info)
            action = ctrl_action.basal + ctrl_action.bolus
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break


if __name__ == '__main__':
    unittest.main()
