This repo contains the code for the implementation of [Distributional Policy Optimization: An Alternative Approach for Continuous Control](https://arxiv.org/abs/1905.09855) (NeurIPS 2019). The theoretical framework is named DPO (Distributional Policy Optimization), whereas the Deep Learning approach to attaining it is named GAC (Generative Actor Critic).

# How to run

An example of how to run the code is provided below. The exact hyper-parameters per each domain are provided in the appendix of the paper.

main.py --visualize --env-name Hopper-v2 --training_actor_samples  32 --noise normal --batch_size 128 --noise_scale 0.2 --print --num_steps 1000000 --target_policy exponential --train_frequency 2048 --replay_size 200000

# Visualizing

You may visualize the run by adding the flag --visualize and starting a visdom server as follows:

python3.6 -m visdom.server

# Requirements

- mujoco - see explanation here: https://github.com/openai/mujoco-py
- gym
- numpy
- tqdm - for tracking experiment time left
- visdom - for visualization of the learning process

# Performance

The graphs below are taken from the paper and compare the performance of our proposed method to various baselines. The best performing method is the Autoregressive network.

![performance graphs](graphs.png?raw=true)
