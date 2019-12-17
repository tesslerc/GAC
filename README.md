# Generative Policy Gradient

for env in Hopper-v2 Swimmer-v2 Walker2d-v2 Humanoid-v2 HalfCheetah-v2 Ant-v2; do python3.6 main.py --visualize --env-name $env --training_actor_samples  32 --noise normal --batch_size 128 --noise_scale 0.2 --print --num_steps 1000000 --target_policy exponential --train_frequency 2048 --replay_size 200000; done

# Starting Visdom

python3.6 -m visdom.server

# Requirements:
- mujoco - see explanation here: https://github.com/openai/mujoco-py
- gym
- numpy
- tqdm - for tracking experiment time left
- visdom - for visualization of the learning process