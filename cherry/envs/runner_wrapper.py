#!/usr/bin/env python3

import cherry as ch
from cherry._utils import _min_size, _istensorable
from .base import Wrapper
from .utils import is_vectorized

from collections.abc import Iterable
from collections import defaultdict


def flatten_episodes(replay, episodes, num_workers):
    """
    TODO: This implementation is not efficient.

    NOTE: Additional info (other than a transition's default fields) is simply copied.
    To know from which worker the data was gathered, you can access sars.runner_id
    TODO: This is not great. What is the best behaviour with infos here ?
    """
    flat_replay = ch.ExperienceReplay()
    worker_replays = [ch.ExperienceReplay() for w in range(num_workers)]
    flat_episodes = 0
    for sars in replay:
        state = sars.state.view(_min_size(sars.state))
        action = sars.action.view(_min_size(sars.action))
        reward = sars.reward.view(_min_size(sars.reward))
        next_state = sars.next_state.view(_min_size(sars.next_state))
        done = sars.done.view(_min_size(sars.done))
        fields = set(sars._Transition__fields) - {'state', 'action', 'reward', 'next_state', 'done'}
        infos = {f: getattr(sars, f) for f in fields}
        for worker in range(num_workers):
            # Populate infos per worker
            worker_infos = {'runner_id': worker}
            for key, value in infos.items():
                worker_infos[key] = value[worker]

            # The following attemps to split additional infos. (WIP. Remove ?)
            # infos = {}
            # for f in fields:
            #     value = getattr(sars, f)
            #     if isinstance(value, Iterable) and len(value) == num_workers:
            #         value = value[worker]
            #     elif _istensorable(value):
            #         tvalue = ch.totensor(value)
            #         tvalue = tvalue.view(_min_size(tvalue))
            #         if tvalue.size(0) == num_workers:
            #             value = tvalue[worker]
            #     infos[f] = value
            worker_replays[worker].append(state[worker],
                                          action[worker],
                                          reward[worker],
                                          next_state[worker],
                                          done[worker],
                                          **worker_infos,
                                          )
            if bool(done[worker]):
                flat_replay += worker_replays[worker]
                worker_replays[worker] = ch.ExperienceReplay()
                flat_episodes += 1
            if flat_episodes >= episodes:
                break
        if flat_episodes >= episodes:
            break
    return flat_replay


class Runner(Wrapper):
    """
    Runner wrapper.

    TODO: When is_vectorized and using episodes=n, use the parallel
    environmnents to sample n episodes, and stack them inside a flat replay.
    """

    def __init__(self, env):
        super(Runner, self).__init__(env)
        self.env = env
        self._needs_reset = True
        self._current_state = None

    def reset(self, *args, **kwargs):
        self._current_state = self.env.reset(*args, **kwargs)
        self._needs_reset = False
        return self._current_state

    def step(self, action, *args, **kwargs):
        # TODO: Implement it to be compatible with .run()
        raise NotImplementedError('Runner does not currently support step.')

    def run(self,
            get_action,
            steps=None,
            episodes=None,
            render=False):
        """
        Runner wrapper's run method.
        """

        if steps is None:
            steps = float('inf')
            if self.is_vectorized:
                self._needs_reset = True
        elif episodes is None:
            episodes = float('inf')
        else:
            msg = 'Either steps or episodes should be set.'
            raise Exception(msg)

        replay = ch.ExperienceReplay()
        collected_episodes = 0
        collected_steps = 0
        while True:
            if collected_steps >= steps or collected_episodes >= episodes:
                if self.is_vectorized and collected_episodes >= episodes:
                    replay = flatten_episodes(replay, episodes, self.num_envs)
                    self._needs_reset = True
                return replay
            if self._needs_reset:
                self.reset()
            info = {}
            action = get_action(self._current_state)
            if isinstance(action, tuple):
                skip_unpack = False
                if self.is_vectorized:
                    if len(action) > 2:
                        skip_unpack = True
                    elif len(action) == 2 and \
                            self.env.num_envs == 2 and \
                            not isinstance(action[1], dict):
                        # action[1] is not info but an action
                        action = (action, )

                if not skip_unpack:
                    if len(action) == 2:
                        info = action[1]
                        action = action[0]
                    elif len(action) == 1:
                        action = action[0]
                    else:
                        msg = 'get_action should return 1 or 2 values.'
                        raise NotImplementedError(msg)
            old_state = self._current_state
            state, reward, done, info = self.env.step(action)
            if not self.is_vectorized and done:
                collected_episodes += 1
                self._needs_reset = True
            elif self.is_vectorized:
                collected_episodes += sum(done)
                # Convert tuple info dictionaries (one per worker) in a single
                # dictionary with a list of values for each key for each worker
                # e.g from this
                # ({key_0: value_0, key_1:value_1}, // worker_0 values
                #  {key_0: value_0, key_1:value_1}) // worker_1 values
                # we get this
                # {key_0: [value_0,  // value of worker 0
                #          value_0], // value of worker 1
                #  key_1: [value_1,  // value of worker 0
                #          value_1], // value of worker 1}
                tmp_info = defaultdict(list)
                for info_worker in info:
                    for key, value in info_worker.items():
                        # Ignore types that cannot be converted to tensors
                        if _istensorable(value):
                            tmp_info[key] += [value]
                info = tmp_info
            replay.append(old_state, action, reward, state, done, **info)
            self._current_state = state
            if render:
                self.env.render()
            collected_steps += 1
