import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from iterative_policy_evaluation import *

policy = initialize_policy(ones=True)
epochs = 1

for _ in tqdm(range(epochs)):
    for s in STATES:
        stable = False
        if s not in [0, 15]:
            values = initialize_values()
            while not stable:
                values = policy_evaluation(policy, values)
                oa = get_next_action(s, policy, vector=False)
                ev = gen_next_action_values(s, values)
                best_action = max(ev, key=ev.get)
                if s in [1, 14]:
                    print(s, best_action)
                stable = oa == best_action
                if not stable:
                    policy[(s, best_action)] += 1
        stable = False

p_map = {}

for s in STATES:
    na = get_next_action(s, policy)
    for k in ACTIONS:
        if ACTIONS[k] == na:
            p_map[s] = k

print(values)
print(p_map)
