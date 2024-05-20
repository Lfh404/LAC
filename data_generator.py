import numpy as np
import torch
from utils import torch_to_numpy


class DataGenerator:
    """
    A data generator used to collect trajectories for on-policy RL
    References:
        https://github.com/Khrylx/PyTorch-RL
        https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
        https://github.com/ikostrikov/pytorch-trpo
    """
    def __init__(self, obs_dim, act_dim, memory_size, batch_size, max_eps_len):

        # Hyperparameters
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.max_eps_len = max_eps_len

        # Batch buffer
        self.obs_buf = np.zeros((memory_size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((memory_size, obs_dim),  dtype=np.float32)
        self.act_buf = np.zeros((memory_size, act_dim),  dtype=np.float32)
        self.rew_buf = np.zeros((memory_size, 1),  dtype=np.float32)
        self.cost_buf = np.zeros((memory_size, 1), dtype=np.float32)
        self.terminal_buf = np.full((memory_size, 1), False, dtype=bool)
        self.buffer_len = 0

        # Episode buffer
        self.obs_eps = np.zeros((max_eps_len, obs_dim),  dtype=np.float32)
        self.next_obs_eps = np.zeros((max_eps_len, obs_dim),  dtype=np.float32)
        self.act_eps = np.zeros((max_eps_len, act_dim),  dtype=np.float32)
        self.rew_eps = np.zeros((max_eps_len, 1),  dtype=np.float32)
        self.cost_eps = np.zeros((max_eps_len, 1), dtype=np.float32)
        self.terminal_eps = np.full((max_eps_len, 1), False, dtype=bool)
        self.eps_len = 0


        # Pointer
        self.ptr = 0

    def run_traj(self, iter, env, policy, nu, actor, critic, critic_cost, running_stat,
                 score_queue, cscore_queue, gamma, c_gamma,
                 dtype, device, constraint, reward_scaling):

        batch_idx = 0

        cost_ret_hist = []
        reward_ret_hist = []

        avg_eps_len = 0
        num_eps = 0
        episode_idx = 0

        # Interact with env for (batch_size) times
        while batch_idx < self.batch_size:
            obs = env.reset()
            if running_stat is not None:
                obs = running_stat.normalize(obs)
            ret_eps = 0
            cost_ret_eps = 0

            for t in range(self.max_eps_len):
                if iter < 5:
                    act = env.action_space.sample()
                else:
                    act = actor.get_act(torch.tensor(obs.reshape(1, *(obs.shape)), dtype=dtype, device=device))
                    act = torch_to_numpy(act).squeeze()
                next_obs, rew, done, info = env.step(act)


                if constraint == 'velocity':
                    if 'y_velocity' not in info:
                        cost = np.abs(info['x_velocity'])
                    else:
                        cost = np.sqrt(info['x_velocity'] ** 2 + info['y_velocity'] ** 2)
                    cost_ret_eps += (c_gamma ** t) * cost
                elif constraint == 'safety': # for safety gym tasks
                    cost = info['cost']
                    cost_ret_eps += cost

                rew *= reward_scaling
                ret_eps += rew

                if running_stat is not None:
                    next_obs = running_stat.normalize(next_obs)

                # Store in episode buffer
                self.obs_eps[t] = obs
                self.act_eps[t] = act
                self.next_obs_eps[t] = next_obs
                self.rew_eps[t] = rew
                self.cost_eps[t] = cost
                self.terminal_eps[t] = done

                obs = next_obs

                batch_idx += 1
                self.eps_len += 1

                # Store return for score if only episode is terminal
                if done or t == self.max_eps_len - 1:
                    score_queue.append(ret_eps / reward_scaling)
                    cscore_queue.append(cost_ret_eps)
                    cost_ret_hist.append(cost_ret_eps)
                    reward_ret_hist.append(ret_eps)

                    num_eps += 1
                    avg_eps_len += (self.eps_len - avg_eps_len) / num_eps

                if done or batch_idx == self.batch_size:
                    break

            # Store episode buffer
            self.obs_eps, self.next_obs_eps = self.obs_eps[:self.eps_len], self.next_obs_eps[:self.eps_len]
            self.act_eps, self.rew_eps = self.act_eps[:self.eps_len], self.rew_eps[:self.eps_len]
            self.cost_eps = self.cost_eps[:self.eps_len]
            self.terminal_eps = self.terminal_eps[:self.eps_len]

            # Update batch buffer
            start_idx, end_idx = self.ptr,  self.ptr + self.eps_len
            end_idx = self.memory_size if end_idx >= self.memory_size else end_idx
            stored_len = end_idx - start_idx

            self.obs_buf[start_idx: end_idx], self.next_obs_buf[start_idx: end_idx] = self.obs_eps[:stored_len], self.next_obs_eps[:stored_len]
            self.act_buf[start_idx: end_idx], self.rew_buf[start_idx: end_idx] = self.act_eps[:stored_len], self.rew_eps[:stored_len]
            self.cost_buf[start_idx: end_idx], self.terminal_buf[start_idx: end_idx] = self.cost_eps[:stored_len], self.terminal_eps[:stored_len]

            # Exceede the buffer memory size
            if end_idx == self.memory_size:
                self.full = True
                start_idx, end_idx = 0, (self.ptr + self.eps_len) % self.memory_size
                self.obs_buf[start_idx: end_idx], self.next_obs_buf[start_idx: end_idx] = self.obs_eps[stored_len:], self.next_obs_eps[stored_len:]
                self.act_buf[start_idx: end_idx], self.rew_buf[start_idx: end_idx] = self.act_eps[stored_len:], self.rew_eps[stored_len:]
                self.cost_buf[start_idx: end_idx], self.terminal_buf[start_idx: end_idx] = self.cost_eps[stored_len:], self.terminal_eps[stored_len:]


            # Update pointer
            self.ptr = end_idx
            
            # Record valid length in memory
            self.buffer_len += self.eps_len
            self.buffer_len = self.buffer_len if self.buffer_len < self.memory_size else self.memory_size

            # Reset episode buffer and update pointer
            self.obs_eps = np.zeros((self.max_eps_len, self.obs_dim), dtype=np.float32)
            self.next_obs_eps = np.zeros((self.max_eps_len, self.obs_dim), dtype=np.float32)
            self.act_eps = np.zeros((self.max_eps_len, self.act_dim), dtype=np.float32)
            self.rew_eps = np.zeros((self.max_eps_len, 1), dtype=np.float32)
            self.cost_eps = np.zeros((self.max_eps_len, 1), dtype=np.float32)
            self.terminal_eps = np.full((self.max_eps_len, 1), False, dtype=bool)
            self.eps_len = 0
            episode_idx += 1


        avg_cost = np.mean(cost_ret_hist)
        std_cost = np.std(cost_ret_hist)
        avg_reward = np.mean(reward_ret_hist) / reward_scaling
        std_reward = np.std(reward_ret_hist) / (reward_scaling**2)


        return {'states':self.obs_buf, 'actions':self.act_buf,
                'next_states':self.next_obs_buf, 'reward':self.rew_buf,
                'cost':self.cost_buf, 'terminals': self.terminal_buf,
                'buffer_len':self.buffer_len, 'avg_cost': avg_cost,
                'std_cost': std_cost, 'avg_eps_len': avg_eps_len,
                'avg_reward': avg_reward, 'std_reward': std_reward}

