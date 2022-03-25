import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from iterative_policy_evaluation import *

def get_action_values():
    pass

policy = initialize_policy(ones=True)
values = initialize_values()

epochs = 1000

for _ in tqdm(range(epochs)):
    values = initialize_values()
    for s in STATES:
        stable = False
        if s not in [0, 15]:
            while not stable:
                values = policy_evaluation(policy, values)
                oa = get_next_action(s, policy, vector=False)
                ev = gen_next_action_values(s, values)
                best_action = max(ev, key=ev.get)
                print(oa, best_action)
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
