import numpy as np
import torch.nn as nn
import torch
import wandb
from torch.distributions import Normal


def mlp(input_size, hidden_sizes=(256, 256), activation='tanh'):

    if activation == 'tanh':
        activation = nn.Tanh
    elif activation == 'relu':
        activation = nn.ReLU
    elif activation == 'sigmoid':
        activation = nn.Sigmoid
    elif activation == 'mish':
        activation = nn.Mish

    layers = []
    sizes = (input_size, ) + hidden_sizes
    for i in range(len(hidden_sizes)):
        layers += [nn.Linear(sizes[i], sizes[i+1]), activation()]
    return nn.Sequential(*layers)



class GaussianPolicy(nn.Module):
    def __init__(self,
                 obs_dim,
                 act_dim,
                 hidden_sizes=(256, 256),
                 activation='tanh',
                 log_std=-0.5,):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.mlp_net = mlp(obs_dim, hidden_sizes, activation)
        self.mean_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.logstd_layer = nn.Parameter(torch.ones(1, act_dim) * log_std)

        self.mean_layer.weight.data.mul_(0.1)
        self.mean_layer.bias.data.mul_(0.0)

    def forward(self, obs):

        out = self.mlp_net(obs)
        mean = self.mean_layer(out)
        if len(mean.size()) == 1:
            mean = mean.view(1, -1)
        logstd = self.logstd_layer.expand_as(mean)
        std = torch.exp(logstd)

        return mean, logstd, std

    def get_act(self, obs, deterministic = False):
        mean, _, std = self.forward(obs)
        if deterministic:
            return mean
        else:
            return torch.normal(mean, std)

    def logprob(self, obs, act):
        mean, _, std = self.forward(obs)
        normal = Normal(mean, std)
        return normal.log_prob(act).sum(-1, keepdim=True), mean, std



class LangevinPolicy(nn.Module):
    def __init__(self,
                 obs_dim,
                 act_dim,
                 max_act,
                 action_scaling,
                 eta=1.0,
                 lam=1.5,
                 langevin_steps=100,
                 langevin_stepsize=0.01,
                 act_grad_norm=40,
                 dtype=torch.float32,
                 device='cuda',
                 wandb=False):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_act = max_act

        self.lam = lam
        self.eta = eta
        self.langevin_steps = langevin_steps
        self.langevin_stepsize = torch.as_tensor(langevin_stepsize, dtype=dtype, device=device)
        self.act_grad_norm = act_grad_norm

        self.wandb = wandb

    def forward(self, obs, reward_critic, cost_critic, nu, actor):
        device = obs.device
        N = obs.shape[0]

        # act = torch.normal(0, 1, size=(N, self.act_dim), device=device)
        act = actor.get_act(obs).detach()
        t = torch.zeros_like(act)

        for i in range(self.langevin_steps):
            act.requires_grad_(True)
            q_reward, q_cost = reward_critic.q_min(obs, act), cost_critic.q_max(obs, act)
            logpi, *_ = actor.logprob(obs, act)
            U = 1. / self.lam * (nu * q_cost - q_reward) - self.eta * logpi
            U.backward(torch.ones_like(U))
            if self.act_grad_norm > 0:
                act_grad_norms = nn.utils.clip_grad_norm_([act], max_norm=self.act_grad_norm, norm_type=2)
                # print(f'act_grad_norms: {act_grad_norms.max()}')

            act.requires_grad_(False)
            act -= self.langevin_stepsize / 2.0 * act.grad
            act += torch.normal(0, 1, size=(N, self.act_dim), device=device) * torch.sqrt(self.langevin_stepsize)
            act.grad.zero_()
            act.clamp_(-self.max_act, self.max_act)


        return act.detach()


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

    def q_max(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.max(q1, q2)
