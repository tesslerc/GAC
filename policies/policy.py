import torch
from torch.autograd import Variable
import os
import numpy as np
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


class Policy:
    def __init__(self, gamma, tau, num_inputs, action_space, replay_size):
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

        self.memory = ReplayMemory(replay_size)
        self.actor = None

    def eval(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def select_action(self, state, action_noise=None):
        state = Variable(state).to(self.device)

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

    def update_critic(self, state_batch, action_batch, reward_batch, mask_batch, next_state_batch):
        raise NotImplementedError

    def update_actor(self, state_batch, action_batch):
        raise NotImplementedError

    def update_parameters(self, batch_size, number_of_iterations):
        policy_losses = []
        value_losses = []

        for _ in range(number_of_iterations):
            transitions = self.memory.sample(batch_size)
            batch = Transition(*zip(*transitions))

            state_batch = Variable(torch.stack(batch.state)).to(self.device)
            action_batch = Variable(torch.stack(batch.action)).to(self.device)
            reward_batch = Variable(torch.stack(batch.reward)).to(self.device).unsqueeze(1)
            mask_batch = Variable(torch.stack(batch.mask)).to(self.device).unsqueeze(1)
            next_state_batch = Variable(torch.stack(batch.next_state)).to(self.device)

            value_loss = self.update_critic(state_batch, action_batch, reward_batch, mask_batch, next_state_batch)
            value_losses.append(value_loss)

            policy_loss = self.update_actor(state_batch, action_batch)
            policy_losses.append(policy_loss)
            self.soft_update()

        return np.mean(value_losses), np.mean(policy_losses)

    def soft_update(self):
        raise NotImplementedError

    def _tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
            self.device)
        return torch.index_select(a, dim, order_index)
