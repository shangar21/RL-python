import numpy as np

class BlackJack():
    CARD_SCORES = {
        'ace':1,
        **{str(i):i for i in range(2,11)},
        'jack':10,
        'queen':10,
        'king':10
    }

    def __init__(self):
        self.DECK = [i for _ in range(4) for i in self.CARD_SCORES]
        self.dealer_cards = self._hit(num_hits=2)
        self.player_cards = self._hit(num_hits=2)
        self.player_stick = False

    def _hit(self, num_hits=1):
        np.random.shuffle(self.DECK)
        return self.DECK[:num_hits]

    def _stick(self):
        self.player_stick = True

    def show_dealer(self):
        return self.dealer_cards[0]

    def _sums(self, cards):
        sums = {'sum':0, 'with_ace_1':0, 'num_ace':0}
        sums['sum'] = sum([self.CARD_SCORES[i] for i in cards])
        sums['num_ace'] = len([self.CARD_SCORES[i] for i in cards if i == 'ace'])
        return sums

    def is_usable_ace(self, cards):
        sum_data = self._sums(cards)
        return sum_data['num_ace'] and sum_data['sum'] + 10 <= 21

    def is_bust(self, cards):
        return self.get_current_sum(cards) >= 21

    def is_blackjack(self, cards):
        return self.get_current_sum(cards) == 21

    def num_usable_ace(self, cards):
        sum_data = self._sums(cards)
        num_potential_ace = (21 - sum_data['sum'])/10

        if self.is_usable_ace(cards):
            if num_potential_ace <= sum_data['num_ace']:
                return num_potential_ace
            return sum_data['num_ace']
        return 0

    def get_current_sum(self, cards):
        sum_data = self._sums(cards)
        if self.is_usable_ace(cards):
            return sum_data['sum'] + (10 * self.num_usable_ace(cards))
        return sum_data['sum']

    def dealer_turn(self):
        while self.get_current_sum(self.dealer_cards) < 17:
            self.dealer_cards += self._hit()

    def reward(self):
        player_cards = self.player_cards
        player_total = self.get_current_sum(player_cards)
        dealer_total = self.get_current_sum(self.dealer_cards)
        if self.is_blackjack(player_cards):
            return 1
        if self.is_bust(self.dealer_cards):
            return 1
        if not self.is_bust(player_cards) and player_total >= dealer_total:
            return 1
        if not self.is_bust(player_cards) and player_total == dealer_total:
            return 0
        if not self.is_bust(player_cards) and player_total <= dealer_total:
            return -1
        return -1

if __name__ == '__main__':
    game = BlackJack()
    print(game.dealer_cards)
    game.dealer_turn()
    print(game.dealer_cards)
    print(game.get_current_sum(game.dealer_cards))
    print(game.is_bust(game.dealer_cards))
