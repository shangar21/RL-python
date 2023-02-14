from Environments.blackjack import BlackJack
from tqdm import tqdm
import multiprocessing as mp

'''
The policy we are evaluating is to hit whenever our current score is less than 20, and otherwise stick
For the first iteration, not considering usable ace, and only looking at the sum of cards at hand
'''

POLICY = {i: {'hit': 1 if i < 20 else 0, 'stick': 0 if i < 20 else 1} for i in range(1, 31)}

#currently state is only the current sum, eventually will make it look at usable ace and dealer card
def get_next_action(policy, state):
    probabilities = policy[state]
    return max(probabilities, key=lambda k: probabilities[k])

# super ugly, will clean this up eventually
def episode(_):
    game = BlackJack()
    rewards = {}
    done = False
    while not done:
        action = get_next_action(POLICY, game.get_current_sum(game.player_cards))
        game.take_action(action)
        if action != 'stick':
            current_sum = game.get_current_sum(game.player_cards)
            rewards[current_sum] = rewards.get(current_sum, []) + [game.reward()]
        else:
            done = True
    game.dealer_turn()
    current_sum = game.get_current_sum(game.player_cards)
    rewards[current_sum] = rewards.get(current_sum, []) + [game.reward()]
    return rewards

def update_dict(original, new):
    for k in new:
        original[k] = original.get(k, []) + new[k]

def first_visit_montecarlo_evaluation(EPOCHS=10_000):
    pool = mp.Pool(mp.cpu_count())
    rewards_record = {}
    for reward in tqdm(pool.imap_unordered(episode, range(EPOCHS)), total=EPOCHS):
        update_dict(rewards_record, reward)
    for k in rewards_record:
        rewards_record[k] = sum(rewards_record[k])/len(rewards_record[k])
    return rewards_record

if __name__ == '__main__':
    rewards_record = first_visit_montecarlo_evaluation()
    print(rewards_record)
