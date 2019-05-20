import torch
from torch.optim import Adam
import torch.nn.functional as F
from network import Critic, Actor
from policies.policy import Policy, hard_update, soft_update


class DDPG(Policy):
    def __init__(self, gamma, tau, num_inputs, action_space, replay_size,
                 normalize_obs=True, normalize_returns=False, critic_l2_reg=1e-2, num_outputs=1, entropy_coeff=0.1,
                 action_coeff=0.1):

        super(DDPG, self).__init__(gamma=gamma, tau=tau, num_inputs=num_inputs, action_space=action_space,
                                   replay_size=replay_size, normalize_obs=normalize_obs,
                                   normalize_returns=normalize_returns)

        self.num_outputs = num_outputs
        self.entropy_coeff = entropy_coeff
        self.action_coeff = action_coeff
        self.critic_l2_reg = critic_l2_reg

        self.actor = Actor(self.num_inputs, self.action_space, self.num_outputs).to(self.device)
        self.actor_target = Actor(self.num_inputs, self.action_space, self.num_outputs).to(self.device)
        self.actor_perturbed = Actor(self.num_inputs, self.action_space, self.num_outputs).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(self.num_inputs + self.action_space.shape[0]).to(self.device)
        self.critic_target = Critic(self.num_inputs + self.action_space.shape[0]).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3, weight_decay=critic_l2_reg)

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def train(self):
        self.actor.train()
        self.critic.train()

    def policy(self, actor, state):
        return actor(state), None

    def update_critic(self, state_batch, action_batch, reward_batch, mask_batch, next_state_batch):
        batch_size = state_batch.shape[0]
        with torch.no_grad():
            tiled_next_state_batch = self._tile(next_state_batch, 0, self.num_outputs)
            next_action_batch, _, next_probs, _ = self.actor_target(next_state_batch)

            next_state_action_values = (self.critic_target(tiled_next_state_batch, next_action_batch.view(batch_size * self.num_outputs, -1))[0].view(batch_size, self.num_outputs) * next_probs).sum(-1).unsqueeze(-1)

            expected_state_action_batch = reward_batch + self.gamma * mask_batch * next_state_action_values

        self.critic_optim.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)[0]
        value_loss = F.mse_loss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        self.critic_optim.step()

        return value_loss.item()

    def update_actor(self, state_batch):
        batch_size = state_batch.shape[0]

        tiled_state_batch = self._tile(state_batch, 0, self.num_outputs)
        action_batch, _, probs, dist_entropy = self.actor(state_batch)

        policy_loss = -(self.critic_target(tiled_state_batch, action_batch.view(batch_size * self.num_outputs, -1))[0].view(batch_size, self.num_outputs) * probs).sum(-1)
        entropy_loss = dist_entropy * self.entropy_coeff

        action_mse = 0
        action_batch = action_batch.view(batch_size, self.num_outputs, -1)
        for idx1 in range(self.num_outputs):
            for idx2 in range(idx1 + 1, self.num_outputs):
                action_mse += ((action_batch[:, idx1, :] - action_batch[:, idx2, :]) ** 2).mean() * self.action_coeff / self.num_outputs

        return policy_loss - entropy_loss - action_mse

    def soft_update(self):
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
