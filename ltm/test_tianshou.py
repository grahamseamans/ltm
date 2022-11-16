import envpool
import numpy as np

train_envs = envpool.make_gym("CartPole-v0", num_envs=10)
test_envs = envpool.make_gym("CartPole-v0", num_envs=100)

