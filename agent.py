from rlsaber.util import compute_v_and_adv
from rollout import Rollout
from build_graph import build_train
import numpy as np
import tensorflow as tf


class Agent:
    def __init__(self,
                 model,
                 actions,
                 optimizer,
                 nenvs,
                 gamma=0.99,
                 lstm_unit=256,
                 time_horizon=5,
                 value_factor=0.5,
                 entropy_factor=0.01,
                 grad_clip=40.0,
                 state_shape=[84, 84, 1],
                 phi=lambda s: s,
                 name='a2c'):
        self.actions = actions
        self.gamma = gamma
        self.name = name
        self.time_horizon = time_horizon
        self.state_shape = state_shape
        self.nenvs = nenvs
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
        states,\
        actions,\
        rewards,\
        values,\
        log_probs,\
        features0,\
        features1,\
        masks = self._rollout_trajectories()

        # compute advantages
        targets = []
        advs = []
        for i in range(self.nenvs):
            v, adv = compute_v_and_adv(
                rewards[i], values[i], bootstrap_values[i], self.gamma)
            targets.append(v)
            advs.append(adv)

        # step size which is usually time horizon
        step_size = len(self.rollouts[0].states)

        # flatten inputs
        states = np.reshape(states, [-1] + self.state_shape)
        actions = np.reshape(actions, [-1])
        targets = np.reshape(targets, [-1])
        advs = np.reshape(advs, [-1])
        log_probs = np.reshape(log_probs, [-1])
        masks = np.reshape(masks, [-1]) == 1.0

        # train network
        loss = self._train(
            states, actions, targets, advs, log_probs,
            features0, features1, masks, step_size)

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
        masks = []
        for rollout in self.rollouts:
            states.append(rollout.states)
            actions.append(rollout.actions)
            rewards.append(rollout.rewards)
            values.append(rollout.values)
            log_probs.append(rollout.log_probs)
            features0.append(rollout.features[0][0])
            features1.append(rollout.features[0][1])
            # create mask
            terminals = rollout.terminals
            terminals = [0.0] + terminals[:len(terminals) - 1]
            mask = (np.array(terminals) - 1.0) * -1.0
            masks.append(mask.tolist())
        return states, actions, rewards, values, log_probs, features0, features1, masks
