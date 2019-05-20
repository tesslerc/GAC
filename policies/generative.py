import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.distributions import Uniform
import torch.nn.functional as F
from network import Critic, StochasticActor, AutoRegressiveStochasticActor
from policies.policy import Policy, hard_update, soft_update


def compute_eltwise_huber_quantile_loss(actions, target_actions, taus, weighting):
    """Compute elementwise Huber losses for quantile regression.
    This is based on Algorithm 1 of https://arxiv.org/abs/1806.06923.
    This function assumes that, both of the two kinds of quantile thresholds,
    taus (used to compute y) and taus_prime (used to compute t) are iid samples
    from U([0,1]).
    Args:
        actions (Variable): Quantile prediction from taus as a
            (batch_size, N, K)-shaped array.
        target_actions (Variable): Quantile targets from taus as a
            (batch_size, N, K)-shaped array.
        taus (ndarray): Quantile thresholds used to compute y as a
            (batch_size, N, 1)-shaped array.
    Returns:
        Variable: Loss
    """
    I_delta = ((actions - target_actions) > 0).float()
    eltwise_huber_loss = F.smooth_l1_loss(actions, target_actions, reduce=False)
    eltwise_loss = abs(taus - I_delta) * eltwise_huber_loss * weighting
    return eltwise_loss.mean()


class Generative(Policy):
    def __init__(self, gamma, tau, num_inputs, action_space, replay_size, normalize_obs=False, normalize_returns=False,
                 num_basis_functions=64, num_outputs=1, use_value=True, q_normalization=0.01, target_policy='linear',
                 target_policy_q='min', autoregressive=True, temp=1.0):

        super(Generative, self).__init__(gamma=gamma, tau=tau, num_inputs=num_inputs, action_space=action_space,
                                         replay_size=replay_size, normalize_obs=normalize_obs,
                                         normalize_returns=normalize_returns)

        self.num_outputs = num_outputs
        self.num_basis_functions = num_basis_functions
        self.action_dim = self.action_space.shape[0]
        self.use_value = use_value
        self.q_normalization = q_normalization
        self.target_policy = target_policy
        self.autoregressive = autoregressive
        self.temp = temp

        if target_policy_q == 'min':
            self.target_policy_q = lambda x, y: torch.min(x, y)
        elif target_policy_q == 'max':
            self.target_policy_q = lambda x, y: torch.max(x, y)
        else:
            self.target_policy_q = lambda x, y: (x + y / 2)

        self.tau_sampler = Uniform(self.Tensor([0.0]), self.Tensor([1.0]))

        '''
            Define networks and optimizers
        '''

        if self.autoregressive:
            self.actor = AutoRegressiveStochasticActor(self.num_inputs, self.action_dim, self.num_basis_functions).to(self.device)
            self.actor_target = AutoRegressiveStochasticActor(self.num_inputs, self.action_dim, self.num_basis_functions).to(self.device)
            self.actor_perturbed = AutoRegressiveStochasticActor(self.num_inputs, self.action_dim, self.num_basis_functions).to(self.device)
        else:
            self.actor = StochasticActor(self.num_inputs, self.action_dim, self.num_basis_functions).to(self.device)
            self.actor_target = StochasticActor(self.num_inputs, self.action_dim, self.num_basis_functions).to(self.device)
            self.actor_perturbed = StochasticActor(self.num_inputs, self.action_dim, self.num_basis_functions).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(self.num_inputs + self.action_dim, num_networks=2).to(self.device)
        self.critic_target = Critic(self.num_inputs + self.action_dim, num_networks=2).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        self.value = Critic(self.num_inputs).to(self.device)
        self.value_target = Critic(self.num_inputs).to(self.device)
        self.value_optim = Adam(self.value.parameters(), lr=1e-3)

        '''
            For multi-gpu setups we enable data parallelism, due to large sample sizes
        '''
        if torch.cuda.device_count() > 1:
            self.actor = torch.nn.DataParallel(self.actor)
            self.actor_target = torch.nn.DataParallel(self.actor_target)
            self.actor_perturbed = torch.nn.DataParallel(self.actor_perturbed)

            self.critic = torch.nn.DataParallel(self.critic)
            self.critic_target = torch.nn.DataParallel(self.critic_target)

            self.value = torch.nn.DataParallel(self.value)
            self.value_target = torch.nn.DataParallel(self.value_target)

        '''
            Initialize target network with the same parameters as the main network
        '''
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
        hard_update(self.value_target, self.value)

    def eval(self):
        self.actor.eval()
        self.critic.eval()
        self.value.eval()

    def train(self):
        self.actor.train()
        self.critic.train()
        self.value.train()

    def policy(self, actor, state, actions=None):
        batch_size = state.shape[0]
        '''
            We sample a quantile for each dimension of the action.
            The action is modeled as an auto-regressive distribution, e.g.,
            P(X) = P(x_0) * P(x_1 | x_0) * ... * P(x_n | x_{n-1}, ..., x_0)
        '''
        taus = self.tau_sampler.rsample((batch_size, self.action_dim)).view(batch_size, self.action_dim, 1)
        return actor(state, taus, actions), None, taus

    def update_critic(self, state_batch, action_batch, reward_batch, mask_batch, next_state_batch):
        batch_size = state_batch.shape[0]

        '''
            Update value network
        '''
        with torch.no_grad():
            # the value is calculated based on multiple samples from the policy and evaluated using the Q networks
            tiled_next_state_batch = self._tile(next_state_batch, 0, self.num_outputs)
            tiled_next_action_batch = self.policy(self.actor_target, tiled_next_state_batch)[0].view(batch_size * self.num_outputs, -1)

            next_q1, next_q2 = self.critic_target(torch.cat((tiled_next_state_batch, tiled_next_action_batch), 1))

            # to avoid over-estimation, we use the minimal value calculated between both Q networks
            next_v = torch.min(
                next_q1.view(batch_size, self.num_outputs).mean(-1).unsqueeze(-1),
                next_q2.view(batch_size, self.num_outputs).mean(-1).unsqueeze(-1)
            )

        if self.use_value:
            v = self.value(next_state_batch)
            value_loss = F.mse_loss(v, next_v)

            self.value_optim.zero_grad()
            value_loss.backward()
            self.value_optim.step()
        else:
            value_loss = torch.Tensor([0])

        '''
            Update Q networks
        '''
        with torch.no_grad():
            if self.use_value:
                next_v = self.value_target(next_state_batch)
            target_q = reward_batch + self.gamma * mask_batch * next_v

        # Add regularization for the Q function. Similar actions should result in similar Q values.
        noise = (self.tau_sampler.rsample((batch_size, self.action_dim)).view(batch_size, self.action_dim) * 2 - 1) * self.q_normalization
        action_batch = (action_batch + noise).clamp(-1, 1)

        q1, q2 = self.critic(torch.cat((state_batch, action_batch), 1))
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        critic_loss = q1_loss + q2_loss

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        return critic_loss.item() + value_loss.item()

    def update_actor(self, state_batch):
        batch_size = state_batch.shape[0]
        tiled_state_batch = self._tile(state_batch, 0, self.num_outputs)

        with torch.no_grad():
            # Calculate the value of each state
            if self.use_value:
                values = self.value_target(state_batch)
            else:
                actions = self.policy(self.actor, tiled_state_batch)[0]
                q1, q2 = self.critic_target(torch.cat((tiled_state_batch, actions), 1))

                # to avoid over-estimation, we use the minimal value calculated between both Q networks
                values = torch.min(
                    q1.view(batch_size, self.num_outputs).mean(-1).unsqueeze(-1),
                    q2.view(batch_size, self.num_outputs).mean(-1).unsqueeze(-1)
                )

            values = torch.cat([values, values], dim=0)

            '''
                Sample multiple actions both from the target policy and from a uniform distribution over the action
                space. These samples are used to compute the target distribution, which is defined as all the actions
                where Q(state, action) > V(state).
            '''
            target_actions = self.policy(self.actor_target, tiled_state_batch)[0]
            target_actions += torch.randn_like(target_actions) * 0.01
            target_actions = target_actions.clamp(-1, 1)

            target_q1, target_q2 = self.critic_target(torch.cat((tiled_state_batch, target_actions), 1))
            target_action_values = self.target_policy_q(
                target_q1.view(batch_size, self.num_outputs, -1),
                target_q2.view(batch_size, self.num_outputs, -1)
            )

            random_actions = torch.rand_like(target_actions) * 2 - 1
            random_q1, random_q2 = self.critic_target(torch.cat((tiled_state_batch, random_actions), 1))
            target_random_values = self.target_policy_q(
                random_q1.view(batch_size, self.num_outputs, -1),
                random_q2.view(batch_size, self.num_outputs, -1)
            )

            target_actions = target_actions.view(batch_size, self.num_outputs, -1)
            random_actions = random_actions.view(batch_size, self.num_outputs, -1)

            target_actions = torch.cat([target_actions, random_actions], dim=0)
            target_action_values = torch.cat([target_action_values, target_random_values], dim=0)

            # (batch_size, 1) -> (batch_size, N, 1)
            values = values.unsqueeze(-1).expand(-1, self.num_outputs, -1)
            improvement = (target_action_values > values).view(-1, 1)  # Choose everything over value

            weighting_improvement = improvement.view(batch_size * 2, self.num_outputs)
            state_improvement = improvement.expand(-1, tiled_state_batch.shape[1])
            action_improvement = improvement.expand(-1, self.action_dim)

            tiled_state_batch = torch.cat([tiled_state_batch, tiled_state_batch], dim=0)
            improving_state_batch = tiled_state_batch[state_improvement].view(-1, tiled_state_batch.shape[1])
            improving_action_batch = target_actions.view(-1, self.action_dim)[action_improvement].view(-1, self.action_dim)

            if self.target_policy == 'linear':
                weighting = (target_action_values[weighting_improvement] - values[weighting_improvement])
                weighting = weighting / weighting.sum(-1, keepdim=True)
            elif self.target_policy == 'boltzman':
                weighting = (target_action_values[weighting_improvement] - values[weighting_improvement])
                weighting = F.softmax((1./self.temp) * weighting, dim=1)
            elif self.target_policy == 'uniform':
                weighting = torch.ones_like(target_action_values[weighting_improvement])
            else: # argmax
                raise NotImplementedError

        if improving_state_batch.shape[0] > 0:
            # Sample multiple actions for each state as an estimation of the current policy
            actions, _, taus = self.policy(self.actor, improving_state_batch, improving_action_batch)
            policy_loss = compute_eltwise_huber_quantile_loss(actions, improving_action_batch, taus.squeeze(-1), weighting)

            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()

            return policy_loss.item()
        else:
            return 0

    def soft_update(self):
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.value_target, self.value, self.tau)
