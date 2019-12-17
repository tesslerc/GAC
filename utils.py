import os
import torch
import numpy as np


def save_model(actor, basedir=None):
    if not os.path.exists('models/'):
        os.makedirs('models/')

    actor_path = "{}/ddpg_actor".format(basedir)
    torch.save(actor.state_dict(), actor_path)


def load_model(agent, basedir=None):
    actor_path = "{}/ddpg_actor".format(basedir)

    print('Loading model from {}'.format(actor_path))
    agent.actor.load_state_dict(torch.load(actor_path))


def moving_average(a, n=3):
    plot_data = np.zeros_like(a)
    for idx in range(len(a)):
        length = min(idx, n)
        plot_data[idx] = a[idx-length:idx+1].mean()
    return plot_data


def vis_plot(viz, log_dict):
    ma_length = 0
    if viz is not None:
        for field in log_dict:
            if len(log_dict[field]) > 0:
                _, values = zip(*log_dict[field])

                plot_data = np.array(log_dict[field])
                viz.line(X=plot_data[:, 0], Y=moving_average(plot_data[:, 1], ma_length), win=field,
                         opts=dict(title=field, legend=[field]))
