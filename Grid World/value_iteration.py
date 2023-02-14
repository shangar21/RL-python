import numpy as np
from iterative_policy_evaluation import *

def get_max_action(state, values):
    action_vals = gen_next_action_values(state, values)
    return action_vals[max(action_vals, key=action_vals.get)]

def value_iteration(values, delta=0):
    for state in STATES:
        if state not in [0, 15]:
            converged = False
            while not converged:
                v = values[state]
                values[state] = get_max_action(state, values)
                converged = np.abs(v-values[state]) <= delta

if __name__ == '__main__':
    values = initialize_values()
    print(values)
    value_iteration(values)
    print(values)
