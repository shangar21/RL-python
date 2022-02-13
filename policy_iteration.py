import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from iterative_policy_evaluation import *

def get_action_values():
    pass

policy = initialize_policy(ones=True)
values = initialize_values()

stable = False
epochs = 1000

for _ in tqdm(range(epochs)):
    while not stable:
        values = policy_evaluation(policy, values)
        for s in STATES:
            if s not in [0, 15]:
                oa = get_next_action(s, policy, vector=False)
                ev = gen_next_action_values(s, values)
                best_action = max(ev, key=ev.get)
                stable = oa == best_action
                if not stable:
                    policy[(s, best_action)] += 1

p_map = {}

for s in STATES:
    na = get_next_action(s, policy)
    for k in ACTIONS:
        if ACTIONS[k] == na:
            p_map[s] = k

print(values)
print(p_map)
