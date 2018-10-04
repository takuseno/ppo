class Rollout:
    def __init__(self):
        self.flush()

    def add(self, obs_t, action_t, reward_tp1, value_t,
            log_prob_t, terminal_tp1=False, feature_t=None):
        self.obs_t.append(obs_t)
        self.actions_t.append(action_t)
        self.rewards_tp1.append(reward_tp1)
        self.values_t.append(value_t)
        self.log_probs_t.append(log_prob_t)
        self.terminals_tp1.append(terminal_tp1)
        self.features_t.append(feature_t)

    def flush(self):
        self.obs_t = []
        self.actions_t = []
        self.rewards_tp1 = []
        self.values_t = []
        self.log_probs_t = []
        self.terminals_tp1 = []
        self.features_t = []
