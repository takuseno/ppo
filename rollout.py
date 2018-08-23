class Rollout:
    def __init__(self):
        self.flush()

    def add(self, state, action, reward, value, log_prob, terminal=False, feature=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.terminals.append(terminal)
        self.features.append(feature)

    def flush(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.terminals = []
        self.features = []
