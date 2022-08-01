from stable_baselines3.ppo.policies import MlpPolicy
from functools import partial
import gym
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from ppo_mem import PPOMemory, ActorCriticMemoryPolicy


def make_env(env_id):
    env = gym.make(env_id)
    # env = gym.wrappers.Monitor(env, f"videos", force=True)  # record videos
    env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
    return env


config = {
    "use_wandb": False,
    "with_memory": False,
    "with_rewards_mem": True,
    "total_timesteps": 100_000,
    "env": "LunarLander-v2",
    # "env": "Humanoid-v3",
    # 'env' : 'HumanoidStandup-v2',
    # 'env' : 'Ant-v2',
    # 'env' : 'Pendulum-v1',
}

if config["use_wandb"]:
    wandb.init(
        config=config,
        sync_tensorboard=True,  # automatically upload SB3's tensorboard metrics to W&B
        project="long term memory lookup",
        monitor_gym=True,  # automatically upload gym environements' videos
        save_code=True,
    )

vec_env = DummyVecEnv([partial(make_env, env_id=config["env"])])

if config["with_memory"]:
    model = PPOMemory(
        ActorCriticMemoryPolicy,
        vec_env,
        verbose=1,
        tensorboard_log=f"runs/ppo",
    )
else:
    model = PPO(
        partial(MlpPolicy, net_arch=[512, 512, 512, dict(vf=[64, 64], pi=[64, 64])]),
        vec_env,
        verbose=1,
        tensorboard_log=f"runs/ppo",
    )

model.learn(total_timesteps=config["total_timesteps"])

if config["use_wandb"]:
    wandb.finish()
