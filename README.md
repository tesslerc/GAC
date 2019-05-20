# Generative Policy Gradient

for env in Hopper-v2 Swimmer-v2 Walker2d-v2 Humanoid-v2 HalfCheetah-v2 Ant-v2; do python3.6 main.py --visualize --env-name $env --policy_type generative --num_outputs 128 --noise normal --batch_size 128 --noise_scale 0.1 --print --number_of_train_steps 100 --num_rollout_steps 100 --num_steps 1000000 --target_policy boltzman --experiment_name boltzmann_128_samples; done

# Starting Visdom

python3.6 -m visdom.server

# Requirements:
- mujoco - see explanation here: https://github.com/openai/mujoco-py
- gym
- numpy
- tqdm - for tracking experiment time left
- visdom - for visualization of the learning process