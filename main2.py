
import argparse
import os
import pprint
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.utils import TensorboardLogger
from tianshou.trainer import offpolicy_trainer

from env import make_env
from policy import DiffusionSAC
from diffusion import Diffusion
from diffusion.model import MLP, DoubleCritic


def get_args():
    parser = argparse.ArgumentParser()

    # common
    parser.add_argument('--task', type=str, default='AaaS')
    parser.add_argument('--algorithm', type=str, default='diffusion_sac')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('-e', '--epoch', type=int, default=1000)
    parser.add_argument('--step-per-epoch', type=int, default=100)
    parser.add_argument('--step-per-collect', type=int, default=1000)
    parser.add_argument('--episode-per-collect', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=512)
    parser.add_argument('--hidden-sizes', type=int, default=256)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log-prefix', type=str, default='default')
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument('--rew-norm', type=int, default=0)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', action='store_true', default=False)
    parser.add_argument('--lr-decay', action='store_true', default=False)
    parser.add_argument('--note', type=str, default='')

    # for diffusion discrete sac
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=0.05)  # for action entropy
    parser.add_argument('--tau', type=float, default=0.005)  # for soft update
    parser.add_argument('-t', '--n-timesteps', type=int, default=5)  # for diffusion chain
    parser.add_argument('--beta-schedule', type=str, default='vp',
                        choices=['linear', 'cosine', 'vp'])
    parser.add_argument('--pg-coef', type=float, default=1.)

    # for prioritized experience replay
    parser.add_argument('--prioritized-replay', action='store_true', default=False)
    parser.add_argument('--prior-alpha', type=float, default=0.6)
    parser.add_argument('--prior-beta', type=float, default=0.4)

    args = parser.parse_known_args()[0]
    return args


def main(args=get_args()):
    # create environments
    env, train_envs, test_envs = make_env(args.task, args.training_num, args.test_num)
    args.state_shape = int(np.prod(env.observation_space.shape))
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = 1.
    print(f'Environment Name: {args.task}')
    print(f'Algorithm Name: DiffusionSAC')
    print(f'Shape of Observation Space: {args.state_shape}')
    print(f'Shape of Action Space: {args.action_shape}')
    # for node in env._swarm_manager._nodes:
    #     print("Node ",node._sid, ",norm_d ", node.norm_total_d)

    env.reset()
    for i in range(100):
        action = np.random.rand(20)
        state, reward, _terminated, _, info  = env.step(action)
        env._swarm_manager.monitor()
        print(reward, _terminated, info)
   
    
if __name__ == '__main__':
    main(get_args())