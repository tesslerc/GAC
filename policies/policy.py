import torch
from torch.autograd import Variable
import os
import numpy as np
from utils import RunningMeanStd
from replay_memory import ReplayMemory, Transition


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def normalize(x, stats, device):
    if stats is None:
        return x
    return (x - torch.Tensor(stats.mean).to(device)) / torch.Tensor(stats.var).sqrt().to(device)


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean


class Policy:
    def __init__(self, gamma, tau, num_inputs, action_space, replay_size, normalize_obs=True,
                 normalize_returns=False):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.enabled = False
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.device = torch.device('cpu')
            self.Tensor = torch.FloatTensor

        self.num_inputs = num_inputs
        self.action_space = action_space

        self.gamma = gamma
        self.tau = tau
        self.normalize_observations = normalize_obs
        self.normalize_returns = normalize_returns

        if self.normalize_observations:
            self.obs_rms = RunningMeanStd(shape=num_inputs)
        else:
            self.obs_rms = None

        if self.normalize_returns:
            self.ret_rms = RunningMeanStd(shape=1)
            self.ret = 0
            self.cliprew = 10.0
        else:
            self.ret_rms = None

        self.memory = ReplayMemory(replay_size)
        self.actor = None
        self.actor_perturbed = None

    def eval(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def select_action(self, state, action_noise=None, param_noise=None):
        state = normalize(Variable(state).to(self.device), self.obs_rms, self.device)

        if param_noise is not None:
            action = self.policy(self.actor_perturbed, state)[0]
        else:
            action = self.policy(self.actor, state)[0]

        action = action.data
        if action_noise is not None:
            action += self.Tensor(action_noise()).to(self.device)

        action = action.clamp(-1, 1)

        return action

    def policy(self, actor, state):
        raise NotImplementedError

    def store_transition(self, state, action, mask, next_state, reward):
        B = state.shape[0]
        for b in range(B):
            self.memory.push(state[b], action[b], mask[b], next_state[b], reward[b])
            if self.normalize_observations:
                self.obs_rms.update(state[b].cpu().numpy())
            if self.normalize_returns:
                self.ret = self.ret * self.gamma + reward[b]
                self.ret_rms.update(np.array([self.ret]))
                if mask[b] == 0:  # if terminal is True
                    self.ret = 0

    def update_critic(self, state_batch, action_batch, reward_batch, mask_batch, next_state_batch):
        raise NotImplementedError

    def update_actor(self, state_batch):
        raise NotImplementedError

    def update_parameters(self, batch_size):
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = normalize(Variable(torch.stack(batch.state)).to(self.device), self.obs_rms, self.device)
        action_batch = Variable(torch.stack(batch.action)).to(self.device)
        reward_batch = normalize(Variable(torch.stack(batch.reward)).to(self.device).unsqueeze(1), self.ret_rms, self.device)
        mask_batch = Variable(torch.stack(batch.mask)).to(self.device).unsqueeze(1)
        next_state_batch = normalize(Variable(torch.stack(batch.next_state)).to(self.device), self.obs_rms, self.device)

        if self.normalize_returns:
            reward_batch = torch.clamp(reward_batch, -self.cliprew, self.cliprew)

        value_loss = self.update_critic(state_batch, action_batch, reward_batch, mask_batch, next_state_batch)
        policy_loss = self.update_actor(state_batch)

        self.soft_update()

        return value_loss, policy_loss

    def soft_update(self):
        raise NotImplementedError

    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name:
                pass
            param = params[name]
            param += torch.randn(param.shape).to(self.device) * param_noise.current_stddev

    def _tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
            self.device)
        return torch.index_select(a, dim, order_index)
