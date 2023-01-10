import argparse
import datetime
import os
import pprint

import gym
import numpy as np
import torch
from mujoco_env import make_mujoco_env
from torch import nn
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer

# from tianshou.policy import PPOPolicy
from ppo_mem import PPOPolicyMem
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.env import SubprocVectorEnv, ShmemVectorEnv, VectorEnvNormObs

from memories import Memories
from mem_nets import Vanilla, SingleThought, MultiThought, Hybrid


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Ant-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=4096)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--mem_len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=30000)
    parser.add_argument("--step-per-collect", type=int, default=2048)
    parser.add_argument("--repeat-per-collect", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--training-num", type=int, default=64)
    parser.add_argument("--test-num", type=int, default=4)
    # ppo special
    parser.add_argument("--rew-norm", type=int, default=True)
    # In theory, `vf-coef` will not make any difference if using Adam optimizer.
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--lazy-coef", type=float, default=0.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--bound-action-method", type=str, default="clip")
    parser.add_argument("--lr-decay", type=int, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=0)
    parser.add_argument("--norm-adv", type=int, default=0)
    parser.add_argument("--recompute-adv", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="mujoco.benchmark")
    parser.add_argument("--wandb-run", type=str, default=None)
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    parser.add_argument("--quickie", type=bool, default=False)
    return parser.parse_args()


def test_ppo(args=get_args()):
    print(f"Using device {args.device}")

    if args.quickie is not False:
        args.epoch = 2
        args.step_per_epoch = 3_000
    env, train_envs, test_envs = make_mujoco_env(
        args.task, args.seed, args.training_num, args.test_num, obs_norm=True
    )
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # model
    memory = Memories(args.state_shape, 1, args.action_shape, args.mem_len)
    net_a = Hybrid(memory)
    actor = ActorProb(
        net_a,
        args.action_shape,
        max_action=args.max_action,
        unbounded=True,
        device=args.device,
    ).to(args.device)
    net_c = Hybrid(memory)
    critic = Critic(net_c, device=args.device).to(args.device)
    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)

    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=args.lr
    )

    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = (
            np.ceil(args.step_per_epoch / args.step_per_collect) * args.epoch
        )

        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
        )

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = PPOPolicyMem(
        memory,
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        action_scaling=True,
        action_bound_method=args.bound_action_method,
        lr_scheduler=lr_scheduler,
        action_space=env.action_space,
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
    )

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "ppo"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    if args.logger == "wandb":
        logger = WandbLogger(
            save_interval=1,
            name=args.wandb_run,
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer)
    else:  # wandb
        logger.load(writer)

        train_envs = ShmemVectorEnv(
            [
                lambda: gym.make(args.task, ctrl_cost_weight=args.lazy_coef)
                for _ in range(args.training_num)
            ]
        )

        # def trigger(x):
        #     return x % args.test_num == 0

        test_envs = SubprocVectorEnv(
            [
                lambda: gym.wrappers.RecordVideo(
                    gym.make(
                        args.task,
                        ctrl_cost_weight=args.lazy_coef,
                        render_mode="rgb_array",
                    ),
                    video_folder=f"videos/{logger.wandb_run.id}",
                    episode_trigger=lambda x: x % args.test_num == 0,
                )
            ]
        )

        train_envs = VectorEnvNormObs(train_envs)
        test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
        test_envs.set_obs_rms(train_envs.get_obs_rms())
        train_envs.seed(args.seed)
        test_envs.seed(args.seed)

    # load a previous policy
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt["model"])
        train_envs.set_obs_rms(ckpt["obs_rms"])
        test_envs.set_obs_rms(ckpt["obs_rms"])
        print("Loaded agent from: ", args.resume_path)

    # collector
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=False)
    test_collector = Collector(policy, test_envs)

    def save_best_fn(policy):
        state = {"model": policy.state_dict()}  # , "obs_rms": train_envs.get_obs_rms()}
        torch.save(state, os.path.join(log_path, "policy.pth"))

    if not args.watch:
        # trainer
        result = onpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.repeat_per_collect,
            args.test_num,
            args.batch_size,
            step_per_collect=args.step_per_collect,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=False,
        )
        pprint.pprint(result)

    # Let's watch its performance!
    # policy.eval()
    # test_envs.seed(args.seed)
    # test_collector.reset()
    # result = test_collector.collect(n_episode=args.test_num, render=args.render)
    # print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')

    def watch():
        policy.eval()
        rec_env = SubprocVectorEnv(
            [
                lambda: gym.wrappers.RecordVideo(
                    gym.make(args.task, render_mode="rgb_array"),
                    video_folder=f"videos/{logger.wandb_run.id}",
                    episode_trigger=lambda x: True,
                )
            ]
        )
        rec_env.seed(args.seed)
        collector = Collector(policy, rec_env, exploration_noise=True)
        collector.collect(n_episode=1)
        collector.collect(n_episode=1)
        collector.collect(n_episode=1)
        collector.collect(n_episode=1)

    if args.watch:
        watch()


if __name__ == "__main__":
    test_ppo()
