import argparse
import os
import gym
import numpy as np
import pickle
from tqdm import trange
import visdom

import torch

from policies.ddpg import DDPG
from policies.generative import Generative
from normalized_actions import NormalizedActions
from noises.ounoise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
from noises.param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
from utils import save_model, vis_plot

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='name of the environment to run')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=5e-3, metavar='G',
                    help='discount factor for model (default: 0.01)')
parser.add_argument('--noise', default='ou', choices=['ou', 'param', 'normal'])
parser.add_argument('--noise_scale', type=float, default=0.2, metavar='G',
                    help='(default: 0.2)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--num_epochs', type=int, default=None, metavar='N',
                    help='number of epochs (default: None)')
parser.add_argument('--num_epochs_cycles', type=int, default=20, metavar='N')
parser.add_argument('--num_rollout_steps', type=int, default=100, metavar='N')
parser.add_argument('--num_steps', type=int, default=2000000, metavar='N',
                    help='number of training steps (default: 2000000)')
parser.add_argument('--param_noise_interval', type=int, default=50, metavar='N')
parser.add_argument('--number_of_train_steps', type=int, default=50, metavar='N')
parser.add_argument('--start_timesteps', type=int, default=10000, metavar='N')
parser.add_argument('--eval_freq', type=int, default=5000, metavar='N')
parser.add_argument('--eval_episodes', type=int, default=10, metavar='N')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--policy_type', default='ddpg', choices=['ddpg', 'generative'])
parser.add_argument('--num_outputs', type=int, default=1, help='Number of outputs for DDPG model.')
parser.add_argument('--visualize', default=False, action='store_true')
parser.add_argument('--experiment_name', default=None, type=str,
                    help='For multiple different experiments, provide an informative experiment name')
parser.add_argument('--print', default=False, action='store_true')
parser.add_argument('--not_autoregressive', default=False, action='store_true')
parser.add_argument('--normalize_obs', default=False, action='store_true', help='Normalize observations')
parser.add_argument('--normalize_rew', default=False, action='store_true', help='Normalize rewards')
parser.add_argument('--q_normalization', type=float, default=0.01,
                    help='Uniformly smooth the Q function in this range.')
parser.add_argument('--target_policy', type=str, default='linear', choices=['linear', 'max', 'boltzman', 'uniform'],
                    help='Target policy is constructed based on this operator.')
parser.add_argument('--target_policy_q', type=str, default='min', choices=['min', 'max', 'mean'],
                    help='The Q value for each sample is determined based on this operator over the two Q networks.')
parser.add_argument('--temp', type=float, default=1.0,
                    help='Boltzman Temperature for normalizing actions')

args = parser.parse_args()

assert args.num_outputs > 0

env = NormalizedActions(gym.make(args.env_name))
eval_env = NormalizedActions(gym.make(args.env_name))

if args.policy_type == 'generative':
    agent = Generative(gamma=args.gamma, tau=args.tau, num_inputs=env.observation_space.shape[0],
                       action_space=env.action_space, replay_size=args.replay_size, num_outputs=args.num_outputs,
                       q_normalization=args.q_normalization, target_policy=args.target_policy,
                       target_policy_q=args.target_policy_q, normalize_obs=args.normalize_obs,
                       normalize_returns=args.normalize_rew, autoregressive=not args.not_autoregressive,
                       temp=args.temp)
else:
    agent = DDPG(gamma=args.gamma, tau=args.tau,  num_inputs=env.observation_space.shape[0],
                 action_space=env.action_space, replay_size=args.replay_size, num_outputs=args.num_outputs,
                 normalize_obs=args.normalize_obs, normalize_returns=args.normalize_rew)

results_dict = {'eval_rewards': [],
                'value_losses': [],
                'policy_losses': [],
                'train_rewards': []
                }
value_losses = []
policy_losses = []

base_dir = os.getcwd() + '/models/' + args.env_name + '/' + args.policy_type + '/'

if args.experiment_name is not None:
    base_dir += args.experiment_name + '/'

if args.policy_type == 'ddpg':
    base_dir += str(args.num_outputs) + '/'

run_number = 0
while os.path.exists(base_dir + str(run_number)):
    run_number += 1
base_dir = base_dir + str(run_number)
os.makedirs(base_dir)

if args.noise == 'ou':
    noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.action_space.shape[0]),
                                         sigma=float(args.noise_scale) * np.ones(env.action_space.shape[0])
                                         )
elif args.noise == 'normal':
    noise = NormalActionNoise(mu=np.zeros(env.action_space.shape[0]),
                              sigma=float(args.noise_scale) * np.ones(env.action_space.shape[0])
                              )
else:
    noise = None

param_noise = AdaptiveParamNoiseSpec(initial_stddev=args.noise_scale,
                                     desired_action_stddev=args.noise_scale) if args.noise == 'param' else None


def reset_noise(a, a_noise, p_noise):
    if a_noise is not None:
        a_noise.reset()
    if p_noise is not None:
        a.perturb_actor_parameters(param_noise)


total_steps = 0
print(base_dir)

if args.num_steps is not None:
    assert args.num_epochs is None
    nb_epochs = int(args.num_steps) // (args.num_epochs_cycles * args.num_rollout_steps)
else:
    nb_epochs = 500


state = agent.Tensor([env.reset()])

episode_reward = 0
agent.train()

reset_noise(agent, noise, param_noise)

if args.visualize:
    vis = visdom.Visdom(env=base_dir)
else:
    vis = None

train_steps = 0
episode_timesteps = 0
for epoch in trange(nb_epochs):
    for cycle in range(args.num_epochs_cycles):
        with torch.no_grad():
            for t_rollout in range(args.num_rollout_steps):
                if total_steps % args.eval_freq == 0:
                    with torch.no_grad():
                        eval_reward = 0
                        for test_epoch in range(args.eval_episodes):
                            done = False
                            eval_state = agent.Tensor([eval_env.reset()])
                            while not done:
                                action = agent.select_action(eval_state)

                                next_eval_state, reward, done, _ = eval_env.step(action.cpu().numpy()[0])
                                eval_reward += reward

                                next_eval_state = agent.Tensor([next_eval_state])

                                eval_state = next_eval_state
                        results_dict['eval_rewards'].append((total_steps, eval_reward * 1.0 / args.eval_episodes))
                        if args.print:
                            try:
                                print('env: {0}, run number: {1}, step: {2}, reward: {3}, value loss: {4}, policy loss: {5}'.format(
                                    args.env_name,
                                    run_number,
                                    results_dict['eval_rewards'][-1][0],
                                    results_dict['eval_rewards'][-1][1],
                                    results_dict['value_losses'][-1][1],
                                    results_dict['policy_losses'][-1][1]))
                            except:
                                pass

                if total_steps < args.start_timesteps:
                    action = torch.Tensor(env.action_space.sample()).to(agent.device).unsqueeze(0)
                else:
                    action = agent.select_action(state, noise, param_noise)
                next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
                done_bool = False if episode_timesteps + 1 == env.env._max_episode_steps else done

                episode_timesteps += 1
                total_steps += 1
                episode_reward += reward

                action = agent.Tensor(action)
                mask = agent.Tensor([not done_bool])
                next_state = agent.Tensor([next_state])
                reward = agent.Tensor([reward])

                agent.store_transition(state, action, mask, next_state, reward)

                state = next_state

                if done:
                    results_dict['train_rewards'].append((total_steps, np.mean(episode_reward)))
                    episode_reward = 0
                    episode_timesteps = 0
                    state = agent.Tensor([env.reset()])
                    reset_noise(agent, noise, param_noise)

        if len(agent.memory) > args.batch_size:
            for t_train in range(args.number_of_train_steps):
                if train_steps % args.param_noise_interval == 0 and param_noise is not None:
                    # Update param_noise based on distance metric
                    episode_transitions = agent.memory.sample(args.batch_size)
                    states = torch.stack([transition[0] for transition in episode_transitions], 0)
                    unperturbed_actions = agent.select_action(states, None, None)
                    perturbed_actions = torch.stack([transition[1] for transition in episode_transitions], 0)

                    ddpg_dist = ddpg_distance_metric(perturbed_actions.cpu().numpy(),
                                                     unperturbed_actions.cpu().numpy())
                    param_noise.adapt(ddpg_dist)

                value_loss, policy_loss = agent.update_parameters(batch_size=args.batch_size)
                value_losses.append(value_loss)
                policy_losses.append(policy_loss)
                train_steps += 1

            results_dict['value_losses'].append((total_steps, np.mean(value_losses)))
            results_dict['policy_losses'].append((total_steps, np.mean(policy_losses)))
            del value_losses[:]
            del policy_losses[:]
        save_model(actor=agent.actor, basedir=base_dir, obs_rms=agent.obs_rms, rew_rms=agent.ret_rms)
        with open(base_dir + '/results', 'wb') as f:
            pickle.dump(results_dict, f)

        vis_plot(vis, results_dict)


with open(base_dir + '/results', 'wb') as f:
    pickle.dump(results_dict, f)
save_model(actor=agent.actor, basedir=base_dir, obs_rms=agent.obs_rms, rew_rms=agent.ret_rms)

env.close()
