import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm

class Machine():
    def __init__(self, num_actions=5, default_mean=[0,1], default_variance=[0,0.1]):
        self.num_actions = num_actions
        self.mu = np.random.uniform(low=default_mean[0], high=default_mean[1], size=num_actions)
        self.sigma = np.random.uniform(low=default_variance[0], high=default_variance[1], size=num_actions)
        self.current_rewards = [0]*num_actions
        self.current_number_of_actions = [0]*num_actions
    
    def try_action(self, action):
        self.current_number_of_actions[action] += 1
        reward = self.reward(action)
        step = 1/self.current_number_of_actions[action]
        error = reward - self.current_rewards[action]
        self.current_rewards[action] += step*error 

    def reward(self, action):
        return np.random.normal(self.mu[action], self.sigma[action], 1)

class NonstationaryMachine():
    def __init__(self, num_actions=5):
        self.num_actions = num_actions
        self.rewards_log = [[0]]*num_actions
        self.current_number_of_actions = [0]*num_actions
        self.rewards = [0]*num_actions
        self.sample_avg_rewards = [0]*num_actions
        self.weighted_avg_rewards = [0]*num_actions 
    
    def sample_average_reward(self, action):
        self.current_number_of_actions[action] += 1
        reward = self.reward(action)
        step = 1/self.current_number_of_actions[action]
        error = reward - self.sample_avg_rewards[action]
        self.sample_avg_rewards[action] += step*error 
    
    def constant_weighted_reward(self, action, alpha):
        reward = self.reward(action)
        self.rewards_log[action].append(reward)
        n = len(self.rewards_log[action])
        p1 = ((1 - alpha)**n)*self.rewards_log[action][0]
        p2 = alpha*sum([((1-alpha)**(n - i))*self.rewards_log[action][i] for i in range(1, len(self.rewards_log[action]))])
        self.weighted_avg_rewards[action] = p1 + alpha*p2

    def reward(self, action):
        self.rewards = list(map(lambda x : x + np.random.normal(0, 1), self.rewards))
        return self.rewards[action]

def action_UCB(rewards, c, t, num_actions):
    sqroot = list(map(lambda x : c*np.sqrt(np.log(t)/x), num_actions))
    vals_arr = [rewards[i] + sqroot[i] for i in range(len(rewards))]
    return np.argmax(vals_arr)

if __name__ == '__main__'

machine = Machine(num_actions=10)
machine_ucb = Machine(num_actions=10)
epsilon = 0.1
steps = 10_000
average_reward = []
average_reward_ucb = []

for _ in range(steps):
    x = np.random.uniform()
    if x > epsilon:
        action = machine.current_rewards.index(max(machine.current_rewards))
    else:
        action = np.random.randint(machine.num_actions)
    action_ucb = action_UCB(machine_ucb.current_rewards, 2, _, machine_ucb.current_number_of_actions)
    machine.try_action(action)
    machine_ucb.try_action(action_ucb)
    average_reward.append(sum(machine.current_rewards)/machine.num_actions)
    average_reward_ucb.append(sum(machine_ucb.current_rewards)/machine_ucb.num_actions)


machine = NonstationaryMachine(num_actions=10)
average_reward_sample_average = []
average_reward_weighted_average = []

for _ in tqdm(range(steps)):
    x = np.random.rand()
    if x > epsilon:
        action_weighted = machine.weighted_avg_rewards.index(max(machine.weighted_avg_rewards))
        action_sample = machine.sample_avg_rewards.index(max(machine.sample_avg_rewards))
    else:
        action_weighted = np.random.randint(machine.num_actions)
        action_sample = np.random.randint(machine.num_actions)
    machine.sample_average_reward(action_sample)
    machine.constant_weighted_reward(action_weighted, 0.1)
    average_reward_sample_average.append(sum(machine.sample_avg_rewards)/machine.num_actions)
    average_reward_weighted_average.append(sum(machine.weighted_avg_rewards)/len(machine.weighted_avg_rewards))

# plt.plot(range(steps), average_reward_sample_average, label='sample mean')
# plt.plot(range(steps), average_reward_weighted_average, label='weighted average')
plt.plot(range(steps), average_reward, label='no UCB')
plt.plot(range(steps), average_reward_ucb, label='ucb')
plt.legend()
plt.show()
