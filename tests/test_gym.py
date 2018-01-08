import gym
import simglucose
import unittest


class TestGym(unittest.TestCase):
    def test_gym_random_agent(self):
        env = gym.make('simglucose-v0')

        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break


if __name__ == '__main__':
    unittest.main()
