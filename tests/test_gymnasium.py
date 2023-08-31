import gymnasium
import unittest
from gymnasium.envs.registration import register


class TestGymnasium(unittest.TestCase):
    def test_gymnasium_random_agent(self):
        register(
            id="simglucose/adolescent2-v0",
            entry_point="simglucose.envs:T1DSimGymnaisumEnv",
            max_episode_steps=10,
            kwargs={"patient_name": "adolescent#002"},
        )

        env = gymnasium.make("simglucose/adolescent2-v0", render_mode="human")
        observation, info = env.reset()
        for t in range(200):
            env.render()
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            print(
                f"Step {t}: observation {observation}, reward {reward}, terminated {terminated}, truncated {truncated}, info {info}"
            )
            if terminated or truncated:
                print("Episode finished after {} timesteps".format(t + 1))
                break


if __name__ == "__main__":
    unittest.main()
