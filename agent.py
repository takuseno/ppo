import network
import build_graph
import lightsaber.tensorflow.util as util
import numpy as np
import tensorflow as tf


class Agent(object):
    def __init__(self, network, obs_dim, num_actions, gamma=0.9):
        self.num_actions = num_actions
        self.gamma = gamma
        self.last_obs = None
        self.t = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []

        act, train, update_old = build_graph.build_train(
            network=network,
            obs_dim=obs_dim,
            num_actions=num_actions,
            gamma=gamma
        )
        self._act = act
        self._train = train
        self._update_old = update_old

    def act(self, obs):
        return self._act([obs])[0]

    def act_and_train(self, obs, reward, episode):
        action, value = self._act([obs])
        action = action[0]
        value = value[0]
        action = np.clip(action, -1, 1)
        reward /= 10.0

        if len(self.states) == 100:
            self.train(value)
            self.states = []
            self.rewards = []
            self.actions = []
            self.values = []

        if self.last_obs is not None:
            self.states.append(self.last_obs)
            self.actions.append(self.last_action)
            self.rewards.append(reward)
            self.values.append(self.last_value)

        self.t += 1
        self.last_obs = obs
        self.last_action = action
        self.last_value = value
        return action

    def train(self, bootstrapped_value):
        returns = []
        deltas = []
        v = bootstrapped_value
        for i in reversed(range(len(self.states))):
            reward = rewards[i]
            v = reward + self.gamma * v
            returns.append(v)
            deltas.append(v - self.values[i])
        returns = np.array(returns, dtype=np.float32)
        deltas = np.array(deltas, dtype=np.float32)
        # standardize advantages
        deltas = (deltas - deltas.mean()) / deltas.std()
        self._train(self.states, self.actions, returns, deltas)

    def stop_episode_and_train(self, obs, reward, done=False):
        self.replay_buffer.append(obs_t=self.last_obs,
                action=self.last_action, reward=reward, obs_tp1=obs, done=done)
        self.stop_episode()

    def stop_episode(self):
        self.last_obs = None
        self.last_action = []
