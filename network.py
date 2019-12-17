import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = self.linear(x)
        return x

    def sample(self, x, deterministic=False):
        x = self(x)

        probs = F.softmax(x, dim=-1)
        if deterministic is False:
            action = probs.multinomial(1)
        else:
            action = probs.max(1)[1]
        return action

    def logprobs_and_entropy(self, x):
        x = self(x)

        log_probs = F.log_softmax(x, dim=-1)
        probs = F.softmax(x, dim=-1)

        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return probs, dist_entropy


class Actor(nn.Module):
    def __init__(self, num_inputs, action_space, num_outputs):
        super(Actor, self).__init__()
        self.action_dim = action_space.shape[0]
        self.num_outputs = num_outputs

        self.common = nn.Sequential(
            nn.Linear(num_inputs, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU()
        )

        self.mu = nn.Linear(300, self.action_dim * self.num_outputs)
        self.dist = Categorical(300, self.num_outputs)

    def forward(self, x):
        common = self.common(x)
        mu = F.tanh(self.mu(common)).view(x.shape[0], self.num_outputs, self.action_dim)
        action = self.dist.sample(common)
        probs, dist_entropy = self.dist.logprobs_and_entropy(x)
        return mu, action, probs, dist_entropy


def cosine_basis_functions(x, n_basis_functions=64):
    x = x.view(-1, 1)
    i_pi = np.tile(np.arange(1, n_basis_functions + 1, dtype=np.float32), (x.shape[0], 1)) * np.pi
    i_pi = torch.Tensor(i_pi)
    if x.is_cuda:
        i_pi = i_pi.cuda()
    embedding = (x * i_pi).cos()
    return embedding


class CosineBasisLinear(nn.Module):
    def __init__(self, n_basis_functions, out_size):
        super(CosineBasisLinear, self).__init__()
        self.linear = nn.Linear(n_basis_functions, out_size)
        self.n_basis_functions = n_basis_functions
        self.out_size = out_size

    def forward(self, x):
        batch_size = x.shape[0]
        h = cosine_basis_functions(x, self.n_basis_functions)
        out = self.linear(h)
        out = out.view(batch_size, -1, self.out_size)
        return out


class AutoRegressiveStochasticActor(nn.Module):
    def __init__(self, num_inputs, action_dim, n_basis_functions):
        super(AutoRegressiveStochasticActor, self).__init__()
        self.action_dim = action_dim
        self.state_embedding = nn.Linear(num_inputs, 400)
        self.noise_embedding = CosineBasisLinear(n_basis_functions, 400)
        self.action_embedding = CosineBasisLinear(n_basis_functions, 400)

        self.rnn = nn.GRU(800, 400, batch_first=True)
        self.l1 = nn.Linear(400, 400)
        self.l2 = nn.Linear(400, 1)

    def forward(self, state, taus, actions=None):
        if actions is not None:
            return self.supervised_forward(state, taus, actions)
        batch_size = state.shape[0]
        # batch x 1 x 400
        state_embedding = F.leaky_relu(self.state_embedding(state)).unsqueeze(1)
        # batch x action dim x 400
        noise_embedding = self.noise_embedding(taus)

        action_list = []

        action = torch.zeros(batch_size, 1)
        if state.is_cuda:
            action = action.cuda()
        hidden_state = None

        for idx in range(self.action_dim):
            # batch x 1 x 400
            action_embedding = F.leaky_relu(self.action_embedding(action.view(batch_size, 1, 1)))
            rnn_input = torch.cat([state_embedding, action_embedding], dim=2)
            gru_out, hidden_state = self.rnn(rnn_input, hidden_state)

            # batch x 400
            hadamard_product = gru_out.squeeze(1) * noise_embedding[:, idx, :]
            action = torch.tanh(self.l2(F.leaky_relu(self.l1(hadamard_product))))
            action_list.append(action)

        actions = torch.stack(action_list, dim=1).squeeze(-1)
        return actions

    def supervised_forward(self, state, taus, actions):
        # batch x action dim x 400
        state_embedding = F.leaky_relu(self.state_embedding(state)).unsqueeze(1).expand(-1, self.action_dim, -1)
        # batch x action dim x 400
        shifted_actions = torch.zeros_like(actions)
        shifted_actions[:, 1:] = actions[:, :-1]
        provided_action_embedding = F.leaky_relu(self.action_embedding(shifted_actions))

        rnn_input = torch.cat([state_embedding, provided_action_embedding], dim=2)
        gru_out, _ = self.rnn(rnn_input)

        # batch x action dim x 400
        noise_embedding = self.noise_embedding(taus)
        # batch x action dim x 400
        hadamard_product = gru_out * noise_embedding
        actions = torch.tanh(self.l2(F.leaky_relu(self.l1(hadamard_product))))
        return actions.squeeze(-1)


class StochasticActor(nn.Module):
    def __init__(self, num_inputs, action_dim, n_basis_functions):
        super(StochasticActor, self).__init__()

        hidden_size = 400

        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.l1 = nn.Linear(num_inputs, self.hidden_size)
        self.phi = CosineBasisLinear(n_basis_functions, self.hidden_size)
        self.l2 = nn.Linear(self.hidden_size, 200)
        self.l3 = nn.Linear(200, self.action_dim)

    def forward(self, state, tau, actions):
        # batch x ~400
        state_embedding = F.leaky_relu(self.l1(state))
        # batch x ~400
        noise_embedding = F.leaky_relu(self.phi(tau)).view(-1, self.hidden_size)

        hadamard_product = state_embedding * noise_embedding

        l2 = F.leaky_relu(self.l2(hadamard_product))

        actions = torch.tanh(self.l3(l2))

        return actions


class Critic(nn.Module):
    def __init__(self, num_inputs, num_networks=1):
        super(Critic, self).__init__()
        self.num_networks = num_networks
        self.q1 = nn.Sequential(
            nn.Linear(num_inputs, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 300),
            nn.LeakyReLU(),
            nn.Linear(300, 1)
        )

        if self.num_networks == 2:
            self.q2 = nn.Sequential(
                nn.Linear(num_inputs, 400),
                nn.LeakyReLU(),
                nn.Linear(400, 300),
                nn.LeakyReLU(),
                nn.Linear(300, 1)
            )
        elif self.num_networks > 2 or self.num_networks < 1:
            raise NotImplementedError

    def forward(self, x):
        if self.num_networks == 1:
            return self.q1(x)
        return self.q1(x), self.q2(x)
