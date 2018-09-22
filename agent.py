from rollout import Rollout
from build_graph import build_train
import numpy as np
import tensorflow as tf


def compute_returns(rewards, bootstrap_value, terminals, gamma):
    # (N, T) -> (T, N)
    rewards = np.transpose(rewards, [1, 0])
    terminals = np.transpose(terminals, [1, 0])
    returns = []
    R = bootstrap_value
    for i in reversed(range(rewards.shape[0])):
        R = rewards[i] + (1.0 - terminals[i]) * gamma * R
        returns.append(R)
    returns = reversed(returns)
    # (T, N) -> (N, T)
    returns = np.transpose(list(returns), [1, 0])
    return returns

def compute_gae(rewards, values, bootstrap_values, terminals, gamma, lam):
    # (N, T) -> (T, N)
    rewards = np.transpose(rewards, [1, 0])
    values = np.transpose(values, [1, 0])
    values = np.vstack((values, [bootstrap_values]))
    terminals = np.transpose(terminals, [1, 0])
    # compute delta
    deltas = []
    for i in reversed(range(rewards.shape[0])):
        V = rewards[i] + (1.0 - terminals[i]) * gamma * values[i + 1]
        delta = V - values[i]
        deltas.append(delta)
    deltas = np.array(list(reversed(deltas)))
    # compute gae
    A = deltas[-1,:]
    advantages = [A]
    for i in reversed(range(deltas.shape[0] - 1)):
        A = deltas[i] + (1.0 - terminals[i]) * gamma * lam * A
        advantages.append(A)
    advantages = reversed(advantages)
    # (T, N) -> (N, T)
    advantages = np.transpose(list(advantages), [1, 0])
    return advantages

class Agent:
    def __init__(self,
                 model,
                 actions,
                 optimizer,
                 nenvs,
                 gamma=0.99,
                 lam=0.95,
                 lstm_unit=256,
                 value_factor=0.5,
                 entropy_factor=0.01,
                 epsilon=0.1,
                 time_horizon=128,
                 batch_size=32,
                 epoch=3,
                 grad_clip=40.0,
                 state_shape=[84, 84, 1],
                 phi=lambda s: s,
                 name='a2c'):
        self.actions = actions
        self.gamma = gamma
        self.lam = lam
        self.lstm_unit = lstm_unit
        self.name = name
        self.state_shape = state_shape
        self.nenvs = nenvs
        self.time_horizon = time_horizon
        self.batch_size = batch_size
        self.epoch = epoch
        self.phi = phi 

        self._act, self._train = build_train(
            model=model,
            num_actions=len(actions),
            optimizer=optimizer,
            nenvs=nenvs,
            lstm_unit=lstm_unit,
            state_shape=state_shape,
            grad_clip=grad_clip,
            value_factor=value_factor,
            entropy_factor=entropy_factor,
            epsilon=epsilon,
            scope=name
        )

        self.initial_state = np.zeros((nenvs, lstm_unit), np.float32)
        self.rnn_state0 = self.initial_state
        self.rnn_state1 = self.initial_state

        self.rollouts = [Rollout() for _ in range(nenvs)]
        self.t = 0

    def act(self, obs_t, reward_t, done_t, training=True):
        # change state shape to WHC
        obs_t = list(map(self.phi, obs_t))
        # take next action
        prob, value, rnn_state = self._act(
            obs_t, self.rnn_state0, self.rnn_state1)
        action_t = list(map(
            lambda p: np.random.choice(range(len(self.actions)), p=p), prob))
        value_t = np.reshape(value, [-1])
        log_probs_t = []
        for i, action, in enumerate(action_t):
            log_probs_t.append(np.log(prob + 1e-20)[i][action])

        self.t += 1
        self.rnn_state0_t = self.rnn_state0
        self.rnn_state1_t = self.rnn_state1
        self.obs_t = obs_t
        self.action_t = action_t
        self.value_t = value_t
        self.log_probs_t = log_probs_t
        self.done_t = done_t
        self.rnn_state0, self.rnn_state1 = rnn_state
        return list(map(lambda a: self.actions[a], action_t))

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
                feature=[self.rnn_state0_t[i], self.rnn_state1_t[i]]
            )

        if update:
            # compute bootstrap value
            _, value, _ = self._act(obs_tp1, self.rnn_state0, self.rnn_state1)
            value_tp1 = np.reshape(value, [-1])
            for i, done in enumerate(done_tp1):
                if done:
                    value_tp1[i] = 0.0
            self.train(value_tp1)

        # initialize lstm state
        for i, done in enumerate(done_tp1):
            if done:
                self.rnn_state0[i] = self.initial_state[0]
                self.rnn_state1[i] = self.initial_state[0]

    def train(self, bootstrap_values):
        # rollout trajectories
        trajectories = self._rollout_trajectories()
        states = trajectories['states']
        actions = trajectories['actions']
        rewards = trajectories['rewards']
        values = trajectories['values']
        log_probs = trajectories['log_probs']
        features0 = trajectories['features0']
        features1 = trajectories['features1']
        terminals = trajectories['terminals']
        masks = trajectories['masks']

        # compute returns
        returns = compute_returns(
            rewards, bootstrap_values, terminals, self.gamma)
        # compute advantages
        advs = compute_gae(
            rewards, values, bootstrap_values, terminals, self.gamma, self.lam)
        # normalize advantages
        valid_advs = np.reshape(advs, [-1])[np.reshape(masks == 1.0, [-1])]
        advs = (advs - np.mean(valid_advs)) / np.std(valid_advs)

        # train network
        for epoch in range(self.epoch):
            for i in range(int(self.time_horizon / self.batch_size)):
                index = i * self.batch_size
                batch_states = self._pick_batch(states, i, shape=self.state_shape)
                batch_actions = self._pick_batch(actions, i)
                batch_returns = self._pick_batch(returns, i)
                batch_advs = self._pick_batch(advs, i)
                batch_log_probs = self._pick_batch(log_probs, i)
                batch_features0 = features0[:, index, :]
                batch_features1 = features1[:, index, :]
                batch_masks = self._pick_batch(masks, i) == 1.0
                loss = self._train(
                    batch_states, batch_actions, batch_returns, batch_advs,
                    batch_log_probs, batch_features0, batch_features1,
                    batch_masks, self.batch_size)

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
        features0 = []
        features1 = []
        terminals = []
        masks = []
        for rollout in self.rollouts:
            states.append(rollout.states)
            actions.append(rollout.actions)
            rewards.append(rollout.rewards)
            values.append(rollout.values)
            log_probs.append(rollout.log_probs)
            features0.append(np.array(rollout.features)[:,0,:])
            features1.append(np.array(rollout.features)[:,1,:])
            # create mask
            terminals.append(rollout.terminals)
            dones = [0.0] + rollout.terminals[:len(rollout.terminals) - 1]
            mask = (np.array(dones) - 1.0) * -1.0
            masks.append(mask.tolist())
        trajectories = dict(
            states=np.array(states),
            actions=np.array(actions),
            rewards=np.array(rewards),
            values=np.array(values),
            log_probs=np.array(log_probs),
            features0=np.array(features0),
            features1=np.array(features1),
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
