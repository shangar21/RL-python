import sys
sys.path.append('../Environments/')
sys.path.append('../Grid World/')
import blackjack
import iterative_policy_evaluation
import numpy as np

policy = iterative_policy_evaluation.initialize_policy()
values = iterative_policy_evaluation.initialize_values()

num_episodes = 1
alpha = 0.1
gamma = 0.9

for i in range(num_episodes):
    s = np.random.randint(1, 15)
    while s != 15:
        a = iterative_policy_evaluation.get_next_action(s, policy, vector=False)
        s_prime = iterative_policy_evaluation.take_next_action(s, a)
        r = -1
        values[s] = values[s] + alpha*(r + gamma*values[s_prime] - values[s])
        s = s_prime

print(values)
