'''
actions:
    up, down, left right
grid world:
    |   | 1 | 2 | 3 |
    | 4 | 5 | 6 | 7 |
    | 8 | 9 | 10| 11|
    | 12| 13| 14|   |
    - blank spots are terminal
S = {1, 2, 3, ..., 14}
A = {up, down, left right}
- undiscounted episodic task
- all actions are equally likely
- policy is that we pick a random one with equal probability for each
- all rewards are -1
- initialize V(s) for all states arbitrarily except V(terminal) = 0
'''

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

ACTIONS = {
    'left': [-1, 0],
    'right': [1, 0],
    'up': [0, -1],
    'down': [0, 1]
}

STATES = list(range(16))

def translate_state(state, grid_length=4):
    # translates state from a number corresponding on grid to a coordinate on grid
    if type(state) == type([]) or type(state) == type(np.array([])):
        return state[0] + state[1]*grid_length
    return [state%grid_length, state//grid_length]

def is_valid_action(state, action):
    return not [i for i in np.add(translate_state(state), ACTIONS[action]) if i < 0 or i > 3] and state not in [0, 15]

def initialize_policy():
    return {(state, action):0.25 if is_valid_action(state, action) else 0 for state in STATES for action in ACTIONS}

def initialize_values():
    return {state: -1 if state not in [0, 14] else 0 for state in STATES}

def get_next_action(state, policy):
    action_probabilities = [policy[(state, k)] for k in ACTIONS]
    if set(action_probabilities) == {0}:
        return None
    choices = np.argwhere(action_probabilities == np.amax(action_probabilities)).flatten()
    return ACTIONS[ACTIONS.keys()[np.random.choice(choices, 1)[0]]]

def gen_valid_next_states(state):
    return [np.add(translate_state(state), ACTIONS[action]) for action in ACTIONS if is_valid_action(state, action)]

def get_new_value(state, policy, values, gamma):
    sum = 0
    for action in ACTIONS:
        if is_valid_action(state, action):
            pi_a_s = policy[(state, action)]
            next_state = translate_state(np.add(translate_state(state), ACTIONS[action]))
            sum += pi_a_s * (-1 + gamma*values[next_state])
    return sum


if __name__ == '__main__':
    policy = initialize_policy()
    values = initialize_values()
    epsilon = 0
    deltas = []
    while epsilon not in deltas:
        for state in STATES:
            v = values[state]
            values[state] = get_new_value(state, policy, values, 0.9)
            deltas.append(v-values[state])
    plt.plot(deltas)
    plt.show()
