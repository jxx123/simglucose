import unittest
from gym.envs.registration import register

register(
    id='simglucose-adolescent1-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={
        'patient_name': 'adolescent#001'
    })


class testRLLab(unittest.TestCase):
    def test_rllab(self):
        try:
            from rllab.algos.ddpg import DDPG
            from rllab.envs.normalized_env import normalize
            from rllab.exploration_strategies.ou_strategy import OUStrategy
            from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
            from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
            from rllab.envs.gym_env import GymEnv
        except ImportError:
            print('rllab is not installed!')
            return None

        env = GymEnv('simglucose-adolescent1-v0')
        env = normalize(env)

        policy = DeterministicMLPPolicy(
            env_spec=env.spec,
            # The neural network policy should have two hidden layers, each
            # with 32 hidden units.
            hidden_sizes=(32, 32))

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
            n_epochs=5,
            discount=0.99,
            scale_reward=0.01,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4)
        algo.train()
