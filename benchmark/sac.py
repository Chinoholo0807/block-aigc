import argparse
import sys
import os
import pprint
import torch
import torch.nn as nn
import numpy as np
from os import path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.policy import DiscreteSACPolicy,SACPolicy
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.trainer import offpolicy_trainer, OffpolicyTrainer

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from env import make_env


def get_args():
    parser = argparse.ArgumentParser()

    # common
    parser.add_argument('--task', type=str, default='AaaS')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--buffer-size', type=int, default=1e6)
    parser.add_argument('--epoch', type=int, default=1e6)
    parser.add_argument('--step-per-epoch', type=int, default=100)
    parser.add_argument('--step-per-collect', type=int, default=1000)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--update-per-step', type=float, default=1)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256, 256])
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log-prefix', type=str, default='default')
    parser.add_argument('--render', type=float, default=0.01)
    parser.add_argument('--rew-norm', type=int, default=0)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', action="store_true", default=False)

    # for sac
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-4)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--auto-alpha', action="store_true", default=False)
    args = parser.parse_known_args()[0]
    return args


def main(args=get_args()):
    # create environment
    env, train_envs, test_envs = make_env(args.task, args.training_num, args.test_num)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    print("device: ", args.device)
    # seed
    if args.task == 'AaaS':
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        train_envs.seed(args.seed)
        test_envs.seed(args.seed)

    # log
    time_now = datetime.now().strftime('%b%d-%H%M%S')
    root = path.dirname(path.dirname(path.abspath(__file__)))
    log_path = os.path.join(
        root, args.logdir, args.log_prefix, args.task, 'sac', time_now)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        if args.reward_threshold:
            return mean_rewards >= args.reward_threshold
        return False
    
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        net_a,
        args.action_shape,
        device=args.device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        action_space=env.action_space,
    )

    # load a previous policy
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt)
        print("Loaded agent from: ", args.resume_path)

    # collector
    # buffer
    buffer = VectorReplayBuffer(
        args.buffer_size,
        buffer_num=len(train_envs)
    )
        
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    # trainer
    if not args.watch:
        result = OffpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            step_per_collect=args.step_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=False,
        ).run()
        pprint.pprint(result)

    # Watch the performance
    if __name__ == '__main__':
        np.random.seed(args.seed)
        env, _, _ = make_env(args.task)
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == '__main__':
    main(get_args())
