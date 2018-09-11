import gym
import unittest
from simglucose.controller.basal_bolus_ctrller import BBController


def custom_reward(BG_last_hour):
    if BG_last_hour[-1] > 180:
        return -1
    elif BG_last_hour[-1] < 70:
        return -2
    else:
        return 1


class TestCustomReward(unittest.TestCase):
    def test_custom_reward(self):
        from gym.envs.registration import register
        register(
            id='simglucose-adolescent3-v0',
            entry_point='simglucose.envs:T1DSimEnv',
            kwargs={
                'patient_name': 'adolescent#003',
                'reward_fun': custom_reward
            })

        env = gym.make('simglucose-adolescent3-v0')
        ctrller = BBController()

        reward = 1
        done = False
        info = {'sample_time': 3, 'patient_name': 'adolescent#002', 'meal': 0}

        observation = env.reset()
        for t in range(200):
            env.render(mode='human')
            print(observation)
            # action = env.action_space.sample()
            ctrl_action = ctrller.policy(observation, reward, done, **info)
            action = ctrl_action.basal + ctrl_action.bolus
            observation, reward, done, info = env.step(action)
            print("Reward = {}".format(reward))
            if observation.CGM > 180:
                self.assertEqual(reward, -1)
            elif observation.CGM < 70:
                self.assertEqual(reward, -2)
            else:
                self.assertEqual(reward, 1)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break


if __name__ == '__main__':
    unittest.main()
