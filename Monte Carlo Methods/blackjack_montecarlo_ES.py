from Environments.blackjack import BlackJack
from blackjack_montecarlo_evaluation import POLICY, episode, get_next_action

def set_policy(policy, state, action_value):
    values = {k: action_value.get((k, state), -float('inf')) for k in BlackJack.ACTIONS}
    best_action = max(values, key=lambda k: values[k])
    policy[state] = {k: 1 if k == best_action else 0 for k in BlackJack.ACTIONS}

def main():
    policy = POLICY
    action_value = {}
    rewards = {}
    game = BlackJack()
    GAMMA = 0.9
    EPOCHS = 100

    for _ in range(EPOCHS):
        G = 0
        episode_log = set()
        done = False
        while not done:
            current_sum = game.get_current_sum(game.player_cards)
            action = get_next_action(policy, current_sum)
            game.take_action(action)
            if action == 'stick':
                game.dealer_turn()
                done = True
            G = GAMMA*G + game.reward()
            state_action_pair = (action, current_sum)
            episode_log.add(state_action_pair)
            if state_action_pair in episode_log:
                rewards[state_action_pair] = rewards.get(state_action_pair, []) + [G]
                action_value[state_action_pair] = sum(rewards[state_action_pair])/len(rewards[state_action_pair])
                set_policy(policy, current_sum, action_value)

    print(policy)
    return policy

if __name__ == '__main__':
    main()
