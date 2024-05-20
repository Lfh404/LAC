import argparse
import os
import wandb
import safety_gym
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
from data_generator import DataGenerator
from models import GaussianPolicy, LangevinPolicy, Critic
from utils import *
from collections import deque
from pathlib import Path


class LAC:
    """
    Implement LAC algorithm
    """
    def __init__(self,
                 env,
                 policy,
                 actor,
                 actor_optimizer,
                 actor_target,
                 reward_critic,
                 cost_critic,
                 reward_critic_target,
                 cost_critic_target,
                 reward_critic_optimizer,
                 cost_critic_optimizer,
                 reward_grad_norm,
                 cost_grad_norm,
                 actor_grad_norm,
                 num_epochs,
                 mb_size,
                 policy_freq,
                 gamma,
                 c_gamma,
                 tau,
                 lam,
                 nu,
                 nu_lr,
                 nu_max,
                 nu_max_factor,
                 cost_lim,
                 l2_reg,
                 score_queue,
                 cscore_queue,
                 logger,
                 wandb=False):


        self.env = env

        self.policy = policy
        self.actor = actor
        self.reward_critic = reward_critic
        self.cost_critic = cost_critic
        self.actor_target = actor_target
        self.reward_critic_target = reward_critic_target
        self.cost_critic_target = cost_critic_target

        self.actor_optimizer = actor_optimizer
        self.reward_critic_optimizer = reward_critic_optimizer
        self.cost_critic_optimizer = cost_critic_optimizer

        self.reward_grad_norm = reward_grad_norm
        self.cost_grad_norm = cost_grad_norm
        self.actor_grad_norm = actor_grad_norm

        self.reward_critic_loss = None
        self.cost_critic_loss = None

        self.num_epochs = num_epochs
        self.mb_size = mb_size
        self.policy_freq = policy_freq

        self.gamma = gamma
        self.c_gamma = c_gamma
        self.tau = tau
        self.lam = lam
        self.cost_lim = cost_lim

        self.nu = nu
        self.nu_lr = nu_lr
        self.nu_max = nu_max
        self.nu_max_factor = nu_max_factor

        self.l2_reg = l2_reg

        self.logger = logger
        self.wandb = wandb
        self.score_queue = score_queue
        self.cscore_queue = cscore_queue


    def update_params(self, iter, rollout, dtype, device):

        # Convert data to tensor
        buffer_len = rollout['buffer_len']
        obs = torch.Tensor(rollout['states'][:buffer_len]).to(dtype)
        act = torch.Tensor(rollout['actions'][:buffer_len]).to(dtype)
        next_obs = torch.Tensor(rollout['next_states'][:buffer_len]).to(dtype)
        reward = torch.Tensor(rollout['reward'][:buffer_len]).to(dtype)
        cost = torch.Tensor(rollout['cost'][:buffer_len]).to(dtype)
        not_done = 1. - torch.Tensor(rollout['terminals'][:buffer_len]).to(dtype)

        # Store in TensorDataset for minibatch updates
        dataset = torch.utils.data.TensorDataset(obs, act, next_obs, reward, cost, not_done)
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.mb_size, shuffle=True)
        avg_cost = rollout['avg_cost']

        reward_critic_loss_list, cost_critic_loss_list = [0.,], [0.,]
        reward_grad_norm_list, cost_grad_norm_list = [0.,], [0.,]
        Q_reward_target_list, Q_cost_target_list = [0.,], [0.,]
        actor_loss_list, actor_grad_norm_list = [0.,], [0.,]


        # Update nu
        self.nu += self.nu_lr * (avg_cost - self.cost_lim)
        if self.nu < 0:
            self.nu = 0
        elif self.nu > self.nu_max:
            self.nu = self.nu_max


        for epoch in range(self.num_epochs):

            for _, (obs_b, act_b, next_obs_b, reward, cost, not_done) in enumerate(loader):

                obs_b, act_b, next_obs_b, reward, cost, not_done = obs_b.to(device), act_b.to(device), next_obs_b.to(device), reward.to(device), cost.to(device), not_done.to(device)

                next_act_b = self.policy(next_obs_b.to(dtype).to(device), self.reward_critic, self.cost_critic, self.nu, self.actor_target)

                """ Q Training """

                # reward critic training
                Q1_reward_cur, Q2_reward_cur = self.reward_critic(obs_b, act_b)
                Q1_reward_target, Q2_reward_target = self.reward_critic_target(next_obs_b, next_act_b)

                Q_reward_target = torch.min(Q1_reward_target, Q2_reward_target)
                Q_reward_target = (reward + not_done * self.gamma * Q_reward_target).detach()
                Q_reward_target_list.append(Q_reward_target.mean().item())

                reward_critic_loss = F.mse_loss(Q1_reward_cur, Q_reward_target) + F.mse_loss(Q2_reward_cur, Q_reward_target)
                reward_critic_loss_list.append(reward_critic_loss.item())
                self.reward_critic_optimizer.zero_grad()
                reward_critic_loss.backward()
                if self.reward_grad_norm > 0:
                    reward_grad_norms = nn.utils.clip_grad_norm_(self.reward_critic.parameters(), max_norm=self.reward_grad_norm, norm_type=2)
                    # print(f"Iter-epoch({iter}-{epoch}): reward_grad_norms({reward_grad_norms.max().item()})", flush=True)
                    reward_grad_norm_list.append(reward_grad_norms.max().item())
                self.reward_critic_optimizer.step()

                # cost critic training
                Q1_cost_cur, Q2_cost_cur = self.cost_critic(obs_b, act_b)
                Q1_cost_target, Q2_cost_target = self.cost_critic_target(next_obs_b, next_act_b)

                Q_cost_target = torch.max(Q1_cost_target, Q2_cost_target)
                Q_cost_target = (cost + not_done * self.c_gamma * Q_cost_target).detach()
                Q_cost_target_list.append(Q_cost_target.mean().item())

                cost_critic_loss = F.mse_loss(Q1_cost_cur, Q_cost_target) + F.mse_loss(Q2_cost_cur, Q_cost_target)
                cost_critic_loss_list.append(cost_critic_loss.item())
                self.cost_critic_optimizer.zero_grad()
                cost_critic_loss.backward()
                if self.cost_grad_norm > 0:
                    cost_grad_norms = nn.utils.clip_grad_norm_(self.cost_critic.parameters(), max_norm=self.cost_grad_norm, norm_type=2)
                    # print(f"Iter-epoch({iter}-{epoch}): cost_grad_norms({cost_grad_norms.max().item()})", flush=True)
                    cost_grad_norm_list.append(cost_grad_norms.max().item())
                self.cost_critic_optimizer.step()

                """ Policy Training"""
                # actor training
                if iter > 0 and iter % self.policy_freq == 0:

                    actor_loss = -self.actor.logprob(next_obs_b, next_act_b)[0].mean()

                    actor_loss_list.append(actor_loss.item())
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    if self.actor_grad_norm > 0:
                        actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_grad_norm, norm_type=2)
                        actor_grad_norm_list.append(actor_grad_norms.max().item())
                    self.actor_optimizer.step()

                    # update the frozen target models
                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                    for param, target_param in zip(self.reward_critic.parameters(), self.reward_critic_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                    for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        Q_reward_target_mean = np.abs(np.array(Q_reward_target_list).mean())
        Q_cost_target_mean = np.abs(np.array(Q_cost_target_list).mean())
        self.nu_max = (self.nu_max_factor * Q_reward_target_mean / Q_cost_target_mean)

        # Store everything in log
        self.logger.update('MinR', np.min(self.score_queue))
        self.logger.update('MaxR', np.max(self.score_queue))
        self.logger.update('AvgR', np.mean(self.score_queue))
        self.logger.update('MinC', np.min(self.cscore_queue))
        self.logger.update('MaxC', np.max(self.cscore_queue))
        self.logger.update('AvgC', np.mean(self.cscore_queue))
        self.logger.update('nu', self.nu)

        if self.wandb:
            wandb.log({
                "reward_critic_loss": np.array(reward_critic_loss_list).mean(),
                "cost_critic_loss": np.array(cost_critic_loss_list).mean(),
                "actor_loss": np.array(actor_loss_list).mean(),
                "Q_reward_target": np.array(Q_reward_target_list).mean(),
                "Q_cost_target": np.array(Q_cost_target_list).mean(),
                "reward_grad_norm": np.array(reward_grad_norm_list).mean(),
                "cost_grad_norm": np.array(cost_grad_norm_list).mean(),
                "actor_grad_norm": np.array(actor_grad_norm_list).mean(),
                "max_reward_grad_norm": np.array(reward_grad_norm_list).max(),
                "max_cost_grad_norm": np.array(cost_grad_norm_list).max(),
                "max_actor_grad_norm": np.array(cost_grad_norm_list).max(),
                "MinR": np.min(self.score_queue),
                "MaxR": np.max(self.score_queue),
                "AvgR": np.mean(self.score_queue),
                "MinC": np.min(self.cscore_queue),
                "MaxC": np.max(self.cscore_queue),
                "AvgC": np.mean(self.cscore_queue),
                "nu": self.nu,
                "nu_max": self.nu_max,
                "epoch": iter,
            })

        # Save models
        self.logger.save_model('reward_critic_params', self.reward_critic_target.state_dict())
        self.logger.save_model('cost_critic_params', self.cost_critic_target.state_dict())
        self.logger.save_model('reward_critic_optimizer', self.reward_critic_optimizer.state_dict())
        self.logger.save_model('cost_critic_optimizer', self.cost_critic_optimizer.state_dict())
        self.logger.save_model('reward_critic_loss', self.reward_critic_loss)
        self.logger.save_model('cost_critic_loss', self.cost_critic_loss)


def train(args):

    # Initialize data type
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Initialize environment
    env = gym.make(args.env_id)
    envname = env.spec.id
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_act = float(env.action_space.high[0])

    # Initialize random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    # Initialize neural nets
    policy = LangevinPolicy(obs_dim,
                            act_dim,
                            max_act,
                            args.eta,
                            lam=args.lam,
                            langevin_steps=args.langevin_steps,
                            langevin_stepsize=args.langevin_stepsize,
                            act_grad_norm=args.act_grad_norm,
                            dtype=dtype,
                            device=device,
                            wandb=args.wandb)
    actor = GaussianPolicy(obs_dim,
                           act_dim,
                           args.hidden_size,
                           args.activation,
                           args.logstd).to(device)
    critic = Critic(obs_dim, act_dim).to(device)
    critic_cost = Critic(obs_dim, act_dim).to(device)
    actor_target = copy.deepcopy(actor)
    critic_target = copy.deepcopy(critic)
    critic_cost_target = copy.deepcopy(critic_cost)


    # Initialize optimizer
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), args.critic_lr)
    critic_cost_optimizer = torch.optim.Adam(critic_cost.parameters(), args.critic_lr)

    # Initialize learning rate scheduler
    lr_lambda = lambda it: max(1.0 - 0.5 * it / args.max_iter_num, 0)
    actor_scheduler = torch.optim.lr_scheduler.LambdaLR(actor_optimizer, lr_lambda=lr_lambda)
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(critic_optimizer, lr_lambda=lr_lambda)
    critic_cost_scheduler = torch.optim.lr_scheduler.LambdaLR(critic_cost_optimizer, lr_lambda=lr_lambda)

    # Store hyperparameters for log
    hyperparams = vars(args)

    # Get constraint bounds
    cost_lim = args.limits

    # Initialize RunningStat for state normalization, score queue, logger
    running_stat = RunningStats(clip=5)
    score_queue = deque(maxlen=100)
    cscore_queue = deque(maxlen=100)
    logger = Logger(hyperparams)
    if args.wandb:
        wandb.init(
            project=args.project_name,
            config={
                "env": args.env_id,
                "constraint": args.constraint,
                "cost-lim": cost_lim,
                "langevin-steps": args.langevin_steps,
                "langevin-stepsize": args.langevin_stepsize,
                "reward-gradnorm": args.reward_grad_norm,
                "cost-gradnorm": args.cost_grad_norm,
                "act-gradnorm": args.act_grad_norm,
                "actor-gradnorm": args.actor_grad_norm,
                "lambda": args.lam,
                "mb-size": args.mb_size,
                "memory-size": args.memory_size,
                "batch-size": args.batch_size,
                "num-epochs": args.num_epochs,
                "max_iter_num": args.max_iter_num,
                "nu_lr": args.nu_lr,
                "nu_max": args.nu_max,
                "eta": args.eta,
                "policy_freq": args.policy_freq,
                "reward_scaling": args.reward_scaling,
            },
            settings=wandb.Settings(
                log_internal=str(os.path.join(os.getcwd(),'wandb','null')),
            )
        )

    # Initialize and train LAC agent
    agent = LAC(env, policy,
                   actor, actor_optimizer,
                   actor_target,
                   critic, critic_cost,
                   critic_target, critic_cost_target,
                   critic_optimizer, critic_cost_optimizer,
                   args.reward_grad_norm, args.cost_grad_norm, args.actor_grad_norm,
                   args.num_epochs, args.mb_size, args.policy_freq,
                   args.gamma, args.c_gamma, args.tau, args.lam,
                   args.nu, args.nu_lr, args.nu_max, args.nu_max_factor, cost_lim,
                   args.l2_reg, score_queue, cscore_queue, logger, args.wandb)

    start_time = time.time()
    data_generator = DataGenerator(obs_dim, act_dim, args.memory_size, args.batch_size, args.max_eps_len)

    for iter in range(args.max_iter_num):

        # Update iteration for model
        agent.logger.save_model('iter', iter)

        # Collect trajectories
        rollout = data_generator.run_traj(iter, env, agent.policy, agent.nu, actor, critic, critic_cost,
                                          running_stat, agent.score_queue, agent.cscore_queue,
                                          args.gamma, args.c_gamma,
                                          dtype, device, args.constraint, args.reward_scaling)

        # Update LAC parameters
        agent.update_params(iter, rollout, dtype, device)

        # Adjust learning rates
        actor_scheduler.step()
        critic_scheduler.step()
        critic_cost_scheduler.step()

        # Update time and running stat
        agent.logger.update('time', time.time() - start_time)
        agent.logger.update('running_stat', running_stat)


        # Save and print values
        print(f'avg_reward: {rollout["avg_reward"]}, avg_cost: {rollout["avg_cost"]}')
        agent.logger.dump()
        if args.wandb:
            wandb.log({"avg_cost": rollout["avg_cost"],
                       "avg_reward": rollout["avg_reward"],
                       "epoch": iter,
                       })

    if args.wandb:
        wandb.finish()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Langevin-Safe-RL Implementation')
    parser.add_argument('--project-name', default='langevin_saferl',
                        help='Name of Project (default: langevin_saferl')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--env-id', default='Safexp-CarGoal1-v0',
                        help='Name of Environment (default: Safexp-CarGoal1-v0')
    parser.add_argument('--constraint', default='safety',
                        help='velocity for MuJoCo-based tasks, safety for others')
    parser.add_argument('--limits', type=float, default=25.0,
                        help='constraint limit for specific env (Default: 25.0)')
    parser.add_argument('--activation', default="tanh",
                        help='Activation function for policy/critic network (Default: tanh)')
    parser.add_argument('--hidden_size', type=float, default=(256, 256),
                        help='Tuple of size of hidden layers for policy network (Default: (256, 256))')
    parser.add_argument('--logstd', type=float, default=-0.5,
                        help='Log std of Policy (Default: -0.5)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for reward (Default: 0.99)')
    parser.add_argument('--c-gamma', type=float, default=0.99,
                        help='Discount factor for cost (Default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='target smoothing coefficient(τ) (Default: 0.005)')
    parser.add_argument('--l2-reg', type=float, default=1e-3,
                        help='L2 Regularization Rate (default: 1e-3)')
    parser.add_argument('--critic-lr', type=float, default=3e-4,
                        help='Learning Rate for critic (default: 3e-4)')
    parser.add_argument('--actor-lr', type=float, default=3e-4,
                        help='Learning Rate for actor (default: 3e-4)')
    parser.add_argument('--critic-cost-lr', type=float, default=3e-4,
                        help='Learning Rate for critic cost (default: 3e-4)')
    parser.add_argument('--reward-grad-norm', type=float, default=0,
                        help='Gradient norm for reward critic (default: 0)')
    parser.add_argument('--cost-grad-norm', type=float, default=0,
                        help='Gradient norm for cost critic (default: 0)')
    parser.add_argument('--act-grad-norm', type=float, default=1000,
                        help='Gradient norm for act in langevin policy (default: 1000)')
    parser.add_argument('--actor-grad-norm', type=float, default=40,
                        help='Gradient norm in Gaussian policy (default: 40)')
    parser.add_argument('--lam', type=float, default=0.0001,
                        help='Inverse temperature lambda (default: 0.0001)')
    parser.add_argument('--nu', type=float, default=0,
                        help='Cost coefficient (default: 0)')
    parser.add_argument('--nu-lr', type=float, default=0.01,
                        help='Cost coefficient learning rate (default: 0.01)')
    parser.add_argument('--nu-max', type=float, default=2.0,
                        help='Maximum cost coefficient (default: 2.0)')
    parser.add_argument('--nu-max-factor', type=float, default=2.0,
                        help='Maximum cost coefficient factor (default: 2.0)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random Seed (default: 0)')
    parser.add_argument('--max-eps-len', type=int, default=1000,
                        help='Maximum length of episode (default: 1000)')
    parser.add_argument('--mb-size', type=int, default=256,
                        help='Minibatch size per update (default: 256)')
    parser.add_argument('--policy-freq', type=int, default=1,
                        help='policy update frequency (default: 1)')
    parser.add_argument('--memory-size', type=int, default=50000,
                        help='memory size in Data Generator (default: 5e5)')
    parser.add_argument('--batch-size', type=int, default=20000,
                        help='Env steps per Update (default: 20000)')
    parser.add_argument('--num-epochs', type=int, default=1,
                        help='Number of passes through each minibatch per update (default: 1)')
    parser.add_argument('--max-iter-num', type=int, default=125,
                        help='Number of Main Iterations (default: 250)')
    parser.add_argument('--langevin-steps', type=int, default=30,
                        help='Number of Langevin policy steps (default: 30)')
    parser.add_argument('--langevin-stepsize', type=float, default=0.0001,
                        help='Stepsize of Langevin policy (default: 0.0001)')
    parser.add_argument('--reward-scaling', type=float, default=10.0,
                        help="scaling for reward (default 10.0)")
    parser.add_argument('--eta', type=float, default=1.,
                        help="weight on old logpi (default 1.)")
    args = parser.parse_args()

    train(args)
