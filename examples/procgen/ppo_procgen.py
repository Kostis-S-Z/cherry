#!/usr/bin/env python3

import os
import json
import datetime
import argparse
import random
import numpy as np

import torch

from procgen import ProcgenEnv
from baselines.common.vec_env import (VecExtractDictObs, VecMonitor, VecNormalize)

from cherry.algorithms.a2c import policy_loss, state_value_loss
# from .models import ActorCritic
# from .sampler import Sampler
from collections import defaultdict


params = {
    "ppo_epochs": 3,
    "lr": 0.0005,
    "backtrack_factor": 0.5,
    "ls_max_steps": 15,
    "max_kl": 0.01,
    "tau": 0.95,
    "gamma": 0.99,

    # Environment params

    # easy or hard: only affects the visual variance between levels
    "distribution_mode": "easy",
    # Number of environments of the same level to run in parallel
    "n_envs": 2,
    # 0-unlimited. For generalization: 200-easy, 500-hard
    "n_levels": 1,
    # Number of total timesteps performed
    "n_timesteps": 5_000_000,
    # Rollout length of each of the above runs
    "n_steps_per_episode": 256,
    # Model params
    "save_every": 100,
    "seed": 42}

# Timesteps performed per task in each iteration
params['steps_per_task'] = int(params['n_steps_per_episode'] * params['n_envs'])
# Split the episode in mini batches
# params['n_mini_batches'] = int(params['steps_per_task'] / params['n_steps_per_mini_batch'])
# iters = outer updates: 64envs, 25M -> 1.525, 200M-> 12.207
params['n_iters'] = int(params['n_timesteps'] // params['steps_per_task'])
# Total timesteps performed per task (if task==1, then total timesteps==total steps per task)
params['total_steps_per_task'] = int(params['steps_per_task'] * params['n_iters'])

net_arch = [64, 128, 128]

env_name = "starpilot"  # Example games: caveflyer, coinrun, dodgeball, maze, starpilot, bigfish
start_level = params['seed']

cuda = True
log_validation = False

path = "results/"
# Create a unique directory for this experiment and save the model's meta-data
model_path = path + 'ppo_' + env_name + '_' + datetime.datetime.now().strftime("%d_%m_%Hh%M")


def compute_a2c_loss(episode_samples, device='cpu'):
    log_prob = torch.from_numpy(episode_samples["log_prob"]).to(device)
    advantages = torch.from_numpy(episode_samples["advantages"]).to(device)
    values = torch.from_numpy(episode_samples["values"]).to(device)
    returns = torch.from_numpy(episode_samples["returns"]).to(device)

    return policy_loss(log_prob, advantages), state_value_loss(values, returns)


def make_procgen_env():
    venv = ProcgenEnv(num_envs=params['n_envs'], env_name=env_name, num_levels=params['n_levels'],
                      start_level=start_level, distribution_mode=params['distribution_mode'],
                      paint_vel_info=True)

    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    return venv


def main():

    device = torch.device('cpu')
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])

    if cuda and torch.cuda.device_count():
        print(f"Running on {torch.cuda.get_device_name(0)}")
        torch.cuda.manual_seed(params['seed'])
        device = torch.device('cuda')

    env = make_procgen_env()

    observ_space = env.observation_space.shape[::-1]
    observ_size = len(observ_space)
    action_space = env.action_space.n

    # Initialize models
    policy = ActorCritic(observ_size, action_space, net_arch)
    policy.to(device)

    actor_optimiser = torch.optim.Adam(policy.actor.parameters(), lr=params['lr'])
    critic_optimiser = torch.optim.Adam(policy.critic.parameters(), lr=params['lr'])

    for iteration in range(params['n_iters']):

        env = make_procgen_env()
        sampler = Sampler(env=env, model=policy, num_steps=params['n_steps_per_episode'],
                          gamma_coef=params['gamma'], lambda_coef=params['tau'],
                          device=device, num_envs=params['n_envs'])

        # Sample training episodes (32envs, 256 length takes less than 1GB)
        tr_ep_samples, tr_ep_infos = sampler.run(no_grad=False, with_adv_ret=False)
        tr_rewards = tr_ep_samples['rewards'].sum().item() / params['n_envs']

        for ppo_epochs in range(params['ppo_epochs']):
            advantages, returns = sampler.calc_adv_ret(tr_ep_samples['rewards'],
                                                       tr_ep_samples['dones'],
                                                       tr_ep_samples['values'])
            tr_ep_samples["advantages"], tr_ep_samples["returns"] = advantages, returns

            actor_loss, critic_loss = compute_a2c_loss(tr_ep_samples, device)

            # Optimize Actor
            actor_optimiser.zero_grad()
            actor_loss.requires_grad = True
            actor_loss.backward()
            actor_optimiser.step()

            # Optimize Critic
            critic_optimiser.zero_grad()
            critic_loss.requires_grad = True
            critic_loss.backward()
            critic_optimiser.step()

        # Average reward across tasks
        tr_iter_reward = tr_rewards
        tr_actor_loss = actor_loss.item()
        tr_critic_loss = critic_loss.item()

        step = iteration * params['n_steps_per_episode'] * params['n_envs']
        metrics = {'tr_iter_reward': tr_iter_reward,
                   'tr_actor_loss': tr_actor_loss,
                   'tr_critic_loss': tr_critic_loss}

        print(f"Step: {step}: metrics: {metrics}")

        if log_validation:
            # Compute validation loss without storing gradients & calculate advantages needed for the loss
            val_ep_samples, val_ep_info = sampler.run(no_grad=True, with_adv_ret=True)
            val_actor_loss, val_critic_loss = compute_a2c_loss(val_ep_samples, device)

            # Update metrics with validation data
            val_iter_reward = val_ep_samples["rewards"].sum().item() / params['n_envs']
            metrics.update({'val_iter_reward': val_iter_reward,
                            'val_actor_loss': val_actor_loss.item(),
                            'val_critic_loss': val_critic_loss.item()})

        if iteration % params['save_every'] == 0:
            save_model_checkpoint(policy, str(iteration))


def save_model_checkpoint(model, epoch):
    save_model(model, name='/model_checkpoints/model_' + epoch)


def save_model(model, name="model"):
    print('Saving ' + name + '...')
    torch.save(model.state_dict(), model_path + '/' + name + '.pt')


""" SAMPLER """


def tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)

    x = np.asarray(x, dtype=np.float)
    x = torch.tensor(x, device=device, dtype=torch.float32)
    return x


def input_preprocessing(x, device):
    x = np.transpose(x, (0, 3, 1, 2))
    x = tensor(x, device)
    x = x.float()
    x /= 255.0
    return x


def to_np(t):
    return t.cpu().detach().numpy()


class Sampler:
    def __init__(self, env, model, num_steps, gamma_coef, lambda_coef, device, num_envs):
        self.env = env
        self.model = model
        self.num_steps = num_steps
        self.lam = lambda_coef
        self.gamma = gamma_coef
        self.device = device

        self.obs = np.zeros(
            (num_envs,) + env.observation_space.shape,
            dtype=env.observation_space.dtype.name,
        )

        self.obs[:] = env.reset()
        self.dones = np.array([False for _ in range(num_envs)])

    def run(self, no_grad=False, with_adv_ret=True):

        # In case you want to collect data without tracking gradients (e.g validation / testing)
        if no_grad:
            with torch.no_grad():
                storage, epinfos = self.collect_experience(with_adv_ret)
        else:
            storage, epinfos = self.collect_experience(with_adv_ret)

        for key in storage:
            if len(storage[key].shape) < 3:
                storage[key] = np.expand_dims(storage[key], -1)
            s = storage[key].shape
            storage[key] = storage[key].swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

        return storage, epinfos

    def collect_experience(self, with_adv_ret):
        # Its a defaultdict and not a dict in order to initialize the default value with a list and append without
        # raising KeyError
        storage = defaultdict(list)  # should contain (state, action, reward, done, next state)
        epinfos = []

        for _ in range(self.num_steps):
            obs = input_preprocessing(self.obs, device=self.device)
            storage["states"] += [to_np(obs.clone())]
            # Forward pass
            prediction, infos = self.model.step(obs)
            actions = to_np(prediction)
            storage["actions"] += [actions]
            storage["values"] += [to_np(infos["value"])]
            storage["log_prob"] += [to_np(infos["log_prob"])]
            storage["mass"] += [infos["mass"]]

            self.obs[:], rewards, self.dones, _ = self.env.step(actions)
            storage["rewards"] += [rewards]
            # Convert booleans to integers
            storage["dones"] += [int(d is True) for d in self.dones]
            storage["next_states"] += [to_np(obs.clone())]
            for info in infos:
                if "episode" in info:
                    epinfos.append(info["episode"])

        # batch of steps to batch of rollouts
        for key in storage:
            storage[key] = np.asarray(storage[key])

        # Calculate PPO's advantages & returns
        if with_adv_ret:
            storage["advantages"], storage["returns"] = self.calc_adv_ret(storage["rewards"],
                                                                          storage["dones"],
                                                                          storage["values"])

        return storage, epinfos

    def calc_adv_ret(self, rewards, dones, values):
        obs = input_preprocessing(self.obs, device=self.device)
        last_values = to_np(self.model.step(obs)[1]["value"])

        # discount/bootstrap
        advantages = np.zeros_like(rewards)

        last_gae_lam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - self.dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]

            td_error = (
                    rewards[t]
                    + self.gamma * next_values * next_non_terminal
                    - values[t]
            )

            advantages[t] = last_gae_lam = (
                    td_error + self.gamma * self.lam * next_non_terminal * last_gae_lam
            )

        returns = advantages + values

        return advantages, returns


""" MODELS """

import math
from collections import OrderedDict

from torch import nn
from torch.distributions import Normal, Categorical

EPSILON = 1e-6


def linear_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()
    return module


def maml_init_(module):
    nn.init.xavier_uniform_(module.weight.data, gain=1.0)
    nn.init.constant_(module.bias.data, 0.0)
    return module


def build_procgen_cnn(input_size, output_size, network):
    n_layers = len(network)
    activation = nn.ReLU

    # Define input layer
    features = OrderedDict(
        {"conv_0": nn.Conv2d(in_channels=input_size, out_channels=network[0], kernel_size=3, padding=1),
         "bn_0": nn.BatchNorm2d(network[0]),
         "activation_0": activation(),
         "max_pool_0": nn.MaxPool2d(kernel_size=2, stride=2)})

    # Initialize weights of input layer
    maml_init_(features["conv_0"])
    nn.init.uniform_(features["bn_0"].weight)

    # Define rest of hidden layers and initialize their weights
    for i in range(1, n_layers):
        layer_i = {f"conv_{i}": nn.Conv2d(in_channels=network[i - 1], out_channels=network[i],
                                          kernel_size=3, stride=1, padding=1),
                   f"bn_{i}": nn.BatchNorm2d(network[i]),
                   f"activation_{i}": activation(),
                   f"max_pool_{i}": nn.MaxPool2d(kernel_size=2, stride=2)}

        maml_init_(layer_i[f"conv_{i}"])
        nn.init.uniform_(layer_i[f"bn_{i}"].weight)
        features.update(layer_i)

    # Given a 64x64 pixel calculate the flatten size needed based on the depth of the network
    # and how "fast" (=stride) it downscales the image
    final_pixel_dim = int(64 / (math.pow(2, n_layers)))
    flatten_size = network[-1] * final_pixel_dim * final_pixel_dim

    network_body = nn.Sequential(*list(features.values()))

    head = nn.Linear(in_features=flatten_size, out_features=output_size)  # No activation for output
    maml_init_(head)

    return flatten_size, network_body, head


class DiagNormalPolicyCNN(nn.Module):

    def __init__(self, input_size, output_size, network=[32, 64, 64]):
        super(DiagNormalPolicyCNN, self).__init__()

        self.flatten_size, self.features, self.mean = build_procgen_cnn(input_size, output_size, network)

        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(1))
        # This is just a trivial assignment to follow the implementation of the sampler
        self.step = self.forward

    def density(self, state):
        # Pass images through CNN to get features
        state = self.features(state)
        # Flatten features to 1-dim for the FC layer
        state = state.view(-1, self.flatten_size)
        # state = state.reshape(-1, self.flatten_size)
        # Pass features to the FC output layer
        loc = self.mean(state)
        scale = torch.exp(torch.clamp(self.sigma, min=math.log(EPSILON)))
        return Normal(loc=loc, scale=scale)

    def log_prob(self, state, action):
        density = self.density(state)
        return density.log_prob(action).mean(dim=1, keepdim=True)

    def forward(self, state):
        density = self.density(state)
        action = density.sample()
        return action


class Actor(nn.Module):
    def __init__(self, input_size, output_size, network, stochastic=True):
        super().__init__()

        self.flatten_size, self.features, self.head = build_procgen_cnn(input_size, output_size, network)

        if stochastic:
            self.policy_log_std = nn.Parameter(torch.tensor([[0.]]))

    def forward(self, state):
        # Pass images through CNN to get features
        features = self.features(state)
        # Flatten features to 1-dim for the FC layer
        features = features.view(-1, self.flatten_size)
        # features = features.reshape(-1, self.flatten_size)
        # Pass features to the FC output layer
        policy = self.head(features)
        return policy


class Critic(nn.Module):
    def __init__(self, input_size, output_size, network, state_action=False):
        super().__init__()
        self.state_action = state_action

        self.flatten_size, self.features, self.head = build_procgen_cnn(input_size, output_size, network)

    def forward(self, state, action=None):
        if self.state_action:
            input_state = torch.cat([state, action], dim=1)
        else:
            input_state = state

        # Pass images through CNN to get features
        features = self.features(input_state)
        # Flatten features to 1-dim for the FC layer
        features = features.view(-1, self.flatten_size)
        # features = features.reshape(-1, self.flatten_size)
        # Pass features to the FC output layer
        value = self.head(features)
        return value.squeeze(dim=1)


class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size, network=[32, 64, 64]):
        super().__init__()
        self.actor = Actor(input_size, output_size, network, stochastic=False)
        self.critic = Critic(input_size, 1, network)

        # This is just a trivial assignment to follow the implementation of the sampler
        self.step = self.forward

    def forward(self, state):
        # policy = Normal(self.actor(state), self.actor.policy_log_std.exp())
        policy = Categorical(logits=self.actor(state))
        value = self.critic(state)
        action = policy.sample()
        log_prob = policy.log_prob(action)
        return action, {
            'mass': policy,
            'log_prob': log_prob,
            'value': value,
        }


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PPO2 on a Procgen env')

    parser.add_argument('--env', type=str, default=env_name, help='Pick an environment')
    parser.add_argument('--lr', type=float, default=params['lr'], help='lr')
    parser.add_argument('--n_iters', type=int, default=params['n_iters'], help='Number of epochs')
    parser.add_argument('--save_every', type=int, default=params['save_every'], help='Interval to save model')
    parser.add_argument('--seed', type=int, default=params['seed'], help='Seed')

    args = parser.parse_args()

    params['lr'] = args.lr
    params['n_iters'] = args.n_iters
    params['save_every'] = args.save_every
    params['seed'] = args.seed

    if not os.path.exists(path):
        os.mkdir(path)
    os.mkdir(model_path)
    os.mkdir(model_path + '/model_checkpoints')

    main()

    with open(model_path + '/parameters.json', 'w') as fp:
        json.dump(params, fp)
