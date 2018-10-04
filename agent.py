import numpy as np
import tensorflow as tf

from rollout import Rollout
from build_graph import build_train
from rlsaber.util import compute_returns, compute_gae


class Agent:
    def __init__(self,
                 model,
                 num_actions,
                 nenvs,
                 lr,
                 epsilon,
                 gamma=0.99,
                 lam=0.95,
                 lstm_unit=256,
                 value_factor=0.5,
                 entropy_factor=0.01,
                 time_horizon=128,
                 batch_size=32,
                 epoch=3,
                 grad_clip=40.0,
                 state_shape=[84, 84, 1],
                 phi=lambda s: s,
                 use_lstm=False,
                 continuous=False,
                 upper_bound=1.0,
                 name='ppo'):
        self.num_actions = num_actions
        self.gamma = gamma
        self.lam = lam
        self.lstm_unit = lstm_unit
        self.name = name
        self.state_shape = state_shape
        self.nenvs = nenvs
        self.lr = lr
        self.epsilon = epsilon
        self.time_horizon = time_horizon
        self.batch_size = batch_size
        self.epoch = epoch
        self.phi = phi 
        self.use_lstm = use_lstm
        self.continuous = continuous
        self.upper_bound = upper_bound

        self._act, self._train = build_train(
            model=model,
            num_actions=num_actions,
            lr=lr.get_variable(),
            epsilon=epsilon.get_variable(),
            nenvs=nenvs,
            step_size=batch_size,
            lstm_unit=lstm_unit,
            state_shape=state_shape,
            grad_clip=grad_clip,
            value_factor=value_factor,
            entropy_factor=entropy_factor,
            continuous=continuous,
            scope=name
        )

        self.initial_state = np.zeros((nenvs, lstm_unit*2), np.float32)
        self.rnn_state = self.initial_state

        self.state_tm1 = dict(obs=None, action=None, value=None,
                              log_probs=None, done=None, rnn_state=None)
        self.rollouts = [Rollout() for _ in range(nenvs)]
        self.t = 0

    def act(self, obs_t, reward_t, done_t, training=True):
        # change state shape to WHC
        obs_t = list(map(self.phi, obs_t))

        # initialize lstm state
        for i, done in enumerate(done_t):
            if done:
                self.rnn_state[i] = self.initial_state[0]

        # take next action
        action_t,\
        log_probs_t,\
        value_t,\
        rnn_state_t = self._act(obs_t, self.rnn_state)
        value_t = np.reshape(value_t, [-1])

        if self.state_tm1['obs'] is not None:
            for i in range(self.nenvs):
                self.rollouts[i].add(
                    obs_t=self.state_tm1['obs'][i],
                    reward_tp1=reward_t[i],
                    action_t=self.state_tm1['action'][i],
                    value_t=self.state_tm1['value'][i],
                    log_prob_t=self.state_tm1['log_probs'][i],
                    terminal_tp1=1.0 if done_t[i] else 0.0,
                    feature_t=self.state_tm1['rnn_state'][i]
                )

        if self.t > 0 and (self.t / self.nenvs) % self.time_horizon == 0:
            bootstrap_values = value_t.copy()
            for i, done in enumerate(self.state_tm1['done']):
                if done:
                    bootstrap_values[i] = 0.0
            self.train(bootstrap_values)

        # decay parameters
        self.t += self.nenvs
        self.lr.decay(self.t)
        self.epsilon.decay(self.t)

        self.rnn_state = rnn_state_t
        self.state_tm1['obs'] = obs_t
        self.state_tm1['action'] = action_t
        self.state_tm1['value'] = value_t
        self.state_tm1['log_probs'] = log_probs_t
        self.state_tm1['done'] = done_t
        self.state_tm1['rnn_state'] = rnn_state_t

        if self.continuous:
            return action_t * self.upper_bound
        else:
            return action_t

    def train(self, bootstrap_values):
        # rollout trajectories
        trajectories = self._rollout_trajectories()
        obs_t = trajectories['obs_t']
        actions_t = trajectories['actions_t']
        rewards_tp1 = trajectories['rewards_tp1']
        values_t = trajectories['values_t']
        log_probs_t = trajectories['log_probs_t']
        features_t = trajectories['features_t']
        terminals_tp1 = trajectories['terminals_tp1']
        masks_t = trajectories['masks_t']

        # compute returns
        returns_t = compute_returns(
            rewards_tp1, bootstrap_values, terminals_tp1, self.gamma)
        # compute advantages
        advs_t = compute_gae(rewards_tp1, values_t, bootstrap_values,
                             terminals_tp1, self.gamma, self.lam)
        # normalize advantages
        advs_t = (advs_t - np.mean(advs_t)) / np.std(advs_t)

        # shuffle batch data if without lstm
        if not self.use_lstm:
            indices = np.random.permutation(range(self.time_horizon))
            obs_t = obs_t[:, indices]
            actions_t = actions_t[:, indices]
            log_probs_t = log_probs_t[:, indices]
            returns_t = returns_t[:, indices]
            advs_t = advs_t[:, indices]
            masks_t = masks_t[:, indices]

        # train network
        for epoch in range(self.epoch):
            for i in range(int(self.time_horizon / self.batch_size)):
                index = i * self.batch_size
                if self.continuous:
                    batch_actions = self._pick_batch(
                        actions_t, i, shape=[self.num_actions])
                    batch_log_probs = self._pick_batch(
                        log_probs_t, i, shape=[self.num_actions])
                else:
                    batch_actions = self._pick_batch(actions_t, i)
                    batch_log_probs = self._pick_batch(log_probs_t, i)
                batch_obs = self._pick_batch(obs_t, i, shape=self.state_shape)
                batch_returns = self._pick_batch(returns_t, i)
                batch_advs = self._pick_batch(advs_t, i)
                batch_features = features_t[:, index, :]
                batch_masks = self._pick_batch(masks_t, i)
                loss = self._train(
                    batch_obs, batch_actions, batch_returns, batch_advs,
                    batch_log_probs, batch_features, batch_masks)

        # clean trajectories
        for rollout in self.rollouts:
            rollout.flush()
        return loss

    def _rollout_trajectories(self):
        obs_t = []
        actions_t = []
        rewards_tp1 = []
        values_t = []
        log_probs_t = []
        features_t= []
        terminals_tp1 = []
        masks_t = []
        for rollout in self.rollouts:
            obs_t.append(rollout.obs_t)
            actions_t.append(rollout.actions_t)
            rewards_tp1.append(rollout.rewards_tp1)
            values_t.append(rollout.values_t)
            log_probs_t.append(rollout.log_probs_t)
            features_t.append(rollout.features_t)
            # create mask
            terminals_tp1.append(rollout.terminals_tp1)
            mask = [0.0] + rollout.terminals_tp1[:self.time_horizon - 1]
            masks_t.append(mask)
        trajectories = dict(
            obs_t=np.array(obs_t),
            actions_t=np.array(actions_t),
            rewards_tp1=np.array(rewards_tp1),
            values_t=np.array(values_t),
            log_probs_t=np.array(log_probs_t),
            features_t=np.array(features_t),
            terminals_tp1=np.array(terminals_tp1),
            masks_t=np.array(masks_t)
        )
        return trajectories

    def _pick_batch(self, data, batch_index, flat=True, shape=None):
        start_index = batch_index * self.batch_size
        batch_data = data[:, start_index:start_index + self.batch_size]
        if flat:
            if shape is not None:
                return np.reshape(batch_data, [-1] + shape)
            return np.reshape(batch_data, [-1])
        else:
            return batch_data
