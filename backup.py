class Backup:
    def __init__(self, num_actors):
        self.num_actors = num_actors
        self.sum_of_rewards = [0 for i in range(num_actors)]
        self.rewards = [0 for i in range(num_actors)]
        self.obs = [0 for i in range(num_actors)]
        self.last_obs = [0 for i in range(num_actors)]
        self.last_actions = [0 for i in range(num_actors)]
        self.last_values = [0 for i in range(num_actors)]
        self.dones = [True for i in range(num_actors)]

    def restore(self, index):
        sum_of_reward = self.sum_of_rewards[index]
        reward = self.rewards[index]
        obs = self.obs[index]
        last_obs = self.last_obs[index]
        last_action = self.last_actions[index]
        last_value = self.last_values[index]
        done = self.dones[index]
        return sum_of_reward, reward, obs,\
                last_obs, last_action, last_value, done

    def save(self, index, sum_of_reward, reward,
            obs, last_obs, last_action, last_value, done):
        self.sum_of_rewards[index] = sum_of_reward
        self.rewards[index] = reward
        self.obs[index] = obs
        self.last_obs[index] = last_obs
        self.last_actions[index] = last_action
        self.last_values[index] = last_value
        self.dones[index] = done
