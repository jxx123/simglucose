import gymnasium as gym

def custom_reward(BG_last_hour):
    if BG_last_hour[-1] > 180:
        return -1
    elif BG_last_hour[-1] < 70:
        return -2
    else:
        return 1


gym.register(
    id='simglucose-adolescent2-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adolescent#002',
            'reward_fun': custom_reward}
)

env = gym.make('simglucose-adolescent2-v0', render_mode='human')

reward = 1

observation, info = env.reset()
for t in range(200):
    env.render()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)
    print("Reward = {}".format(reward))
    if terminated or truncated:
        print("Episode finished after {} timesteps".format(t + 1))
        break

env.close()