#!/usr/bin/env python3

import random
import gym
import numpy as np
import randopt as ro

import torch as th
import torch.optim as optim
import torch.nn as nn

import cherry as ch
import cherry.envs as envs
import cherry.mpi as mpi

from statistics import mean

from models import NatureCNN

"""
This is a demonstration of how to use cherry to train an agent in a distributed
setting.

TODO:
    * for last 1000 steps, the reward count and lengths are off.
"""

GAMMA = 0.99
USE_GAE = False
TAU = 0.95
V_WEIGHT = 0.5
ENT_WEIGHT = 0.01
LR = 1e-4
GRAD_NORM = 5.0
NUM_UPDATES = 0
A2C_STEPS = 5


def update(replay, optimizer, policy, env=None):
    values = [info['value'] for info in replay.infos]

    # Discount and normalize rewards
    if USE_GAE:
        _, next_state_value = policy(replay.next_states[-1])
        advantages = ch.rewards.gae(GAMMA,
                                    TAU,
                                    replay.rewards,
                                    replay.dones,
                                    replay.values,
                                    next_state_value)
    rewards = ch.rewards.discount(GAMMA,
                                  replay.rewards,
                                  replay.dones,
                                  bootstrap=values[-1])
    if not USE_GAE:
        advantages = [r - v for r, v in zip(rewards, values)]

    # Compute losses
    policy_loss = 0.0
    value_loss = 0.0
    entropy_loss = 0.0
    for info, reward, adv in zip(replay.infos, rewards, advantages):
        entropy_loss += - info['entropy']
        policy_loss += -info['log_prob'] * adv.item()
        value_loss += 0.5 * (reward.detach() - info['value'])**2

    # Take optimization step
    optimizer.zero_grad()
    loss = policy_loss + V_WEIGHT * value_loss + ENT_WEIGHT * entropy_loss
    loss /= len(rewards)
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), GRAD_NORM)
    optimizer.step()

    global NUM_UPDATES
    NUM_UPDATES += 1
    if env is not None:
        env.log('policy loss', policy_loss.item())
        env.log('value loss', value_loss.item())
        env.log('entropy', entropy_loss.item())
        env.log('num updates', NUM_UPDATES)


def get_action_value(state, policy):
    mass, value = policy(state)
    action = mass.sample()
    info = {
        'log_prob': mass.log_prob(action),  # Cache log_prob for later
        'value': value,
        'entropy': mass.entropy(),
    }
    return action, info


@ro.cli
def main(num_steps=10000000,
         env='PongNoFrameskip-v4',
         seed=1234):
    rank = mpi.rank
    th.set_num_threads(1)
    random.seed(seed + rank)
    th.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    env = gym.make(env)
    if rank == 0:
        env = envs.Logger(env, interval=1000)
    env = envs.OpenAIAtari(env)
    env = envs.Torch(env)
    env = envs.Runner(env)
    env.seed(seed + rank)

    policy = NatureCNN(num_outputs=env.action_size)
    optimizer = optim.RMSprop(policy.parameters(), lr=1e-4, alpha=0.99, eps=1e-5)
    optimizer = mpi.Distributed(policy.parameters(), optimizer)
    replay = ch.ExperienceReplay()
    get_action = lambda state: get_action_value(state, policy)

    total_num_steps = num_steps
    total_steps = 0
    env.seed(1234)
    while total_steps < total_num_steps:
        # Sample some transitions
        num_steps, num_episodes = env.run(get_action, replay, steps=A2C_STEPS)

        # Update policy
        update(replay, optimizer, policy, env=env)
        replay.empty()
        if rank == 0:
            total_steps += num_steps

    # Save results with randopt
    if rank == 0:
        exp = ro.Experiment('dev', directory='results')
        rewards = env.all_rewards
        dones = env.all_dones
        result = mean(rewards[-1000:])
        data = {
                'rewards': rewards,
                'dones': dones,
                }
        exp.add_result(result, data)
        # Kill all MPI processes
        mpi.comm.Abort()


if __name__ == '__main__':
    ro.parse()
