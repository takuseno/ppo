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

        self.rollouts = [Rollout() for _ in range(nenvs)]
        self.t = 0

    def act(self, obs_t, reward_t, done_t, training=True):
        # change state shape to WHC
        obs_t = list(map(self.phi, obs_t))
        # take next action
        action_t, log_probs_t, value, rnn_state = self._act(obs_t, self.rnn_state)
        value_t = np.reshape(value, [-1])

        self.t += self.nenvs
        self.rnn_state_t = self.rnn_state
        self.obs_t = obs_t
        self.action_t = action_t
        self.value_t = value_t
        self.log_probs_t = log_probs_t
        self.done_t = done_t
        self.rnn_state = rnn_state

        # decay parameters
        self.lr.decay(self.t)
        self.epsilon.decay(self.t)

        if self.continuous:
            return action_t * self.upper_bound
        else:
            return action_t

    # this method is called after act
    def receive_next(self, obs_tp1, reward_tp1, done_tp1, update=False):
        obs_tp1 = list(map(self.phi, obs_tp1))

        for i in range(self.nenvs):
            self.rollouts[i].add(
                state=self.obs_t[i],
                reward=reward_tp1[i],
                action=self.action_t[i],
                value=self.value_t[i],
                log_prob=self.log_probs_t[i],
                terminal=1.0 if done_tp1[i] else 0.0,
                feature=self.rnn_state_t[i]
            )

        if update:
            # compute bootstrap value
            _, _, value, _ = self._act(obs_tp1, self.rnn_state)
            value_tp1 = np.reshape(value, [-1])
            for i, done in enumerate(done_tp1):
                if done:
                    value_tp1[i] = 0.0
            self.train(value_tp1)

        # initialize lstm state
        for i, done in enumerate(done_tp1):
            if done:
                self.rnn_state[i] = self.initial_state[0]

    def train(self, bootstrap_values):
        # rollout trajectories
        trajectories = self._rollout_trajectories()
        states = trajectories['states']
        actions = trajectories['actions']
        rewards = trajectories['rewards']
        values = trajectories['values']
        log_probs = trajectories['log_probs']
        features = trajectories['features']
        terminals = trajectories['terminals']
        masks = trajectories['masks']

        # compute returns
        returns = compute_returns(
            rewards, bootstrap_values, terminals, self.gamma)
        # compute advantages
        advs = compute_gae(
            rewards, values, bootstrap_values, terminals, self.gamma, self.lam)
        # normalize advantages
        advs = (advs - np.mean(advs)) / np.std(advs)

        # shuffle batch data if without lstm
        if not self.use_lstm:
            indices = np.random.permutation(range(self.time_horizon))
            states = np.array(states)[:, indices]
            actions = np.array(actions)[:, indices]
            log_probs = np.array(log_probs)[:, indices]
            returns = np.array(returns)[:, indices]
            advs = np.array(advs)[:, indices]
            masks = np.array(masks)[:, indices]

        # train network
        for epoch in range(self.epoch):
            for i in range(int(self.time_horizon / self.batch_size)):
                index = i * self.batch_size
                if self.continuous:
                    batch_actions = self._pick_batch(
                        actions, i, shape=[self.num_actions])
                    batch_log_probs = self._pick_batch(
                        log_probs, i, shape=[self.num_actions])
                else:
                    batch_actions = self._pick_batch(actions, i)
                    batch_log_probs = self._pick_batch(log_probs, i)
                batch_states = self._pick_batch(
                    states, i, shape=self.state_shape)
                batch_returns = self._pick_batch(returns, i)
                batch_advs = self._pick_batch(advs, i)
                batch_features = features[:, index, :]
                batch_masks = self._pick_batch(masks, i)
                loss = self._train(
                    batch_states, batch_actions, batch_returns, batch_advs,
                    batch_log_probs, batch_features, batch_masks)

        # clean trajectories
        for rollout in self.rollouts:
            rollout.flush()
        return loss

    def _rollout_trajectories(self):
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        features = []
        terminals = []
        masks = []
        for rollout in self.rollouts:
            states.append(rollout.states)
            actions.append(rollout.actions)
            rewards.append(rollout.rewards)
            values.append(rollout.values)
            log_probs.append(rollout.log_probs)
            features.append(rollout.features)
            # create mask
            terminals.append(rollout.terminals)
            mask = [0.0] + rollout.terminals[:len(rollout.terminals) - 1]
            masks.append(mask)
        trajectories = dict(
            states=np.array(states),
            actions=np.array(actions),
            rewards=np.array(rewards),
            values=np.array(values),
            log_probs=np.array(log_probs),
            features=np.array(features),
            terminals=np.array(terminals),
            masks=np.array(masks)
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
