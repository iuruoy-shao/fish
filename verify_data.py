import copy
import json
import numpy as np
import re

with open('sets.json','r') as f:
    sets = np.array(json.load(f))
    card_to_vector = {}
    for card in sets.flatten():
        vector = np.zeros(15, int)
        vector[np.where(sets == card)[0][0]] = 1
        vector[np.where(sets == card)[1][0] + 9] = 1
        card_to_vector[card] = vector

class ParseError(Exception):
    pass

class FishGame:
    def __init__(self, datarows):
        self.datarows = datarows
        self.init_hands = {player.split(":")[0]:set(player.split(":")[1][1:-1].split(',')) for player in self.datarows[0].split()}
        self.players = list(self.init_hands.keys())
        self.score = [0, 0]
        self.rewards = [] # encodes for the odd team, reverse for even team
        self.verify()
        self.to_state()

    def initials_to_index(self, initials):
        return self.players.index(initials)

    def teammates(self, initials):
        return self.players[self.initials_to_index(initials) % 2::2]

    def parse_call(self, line):
        return {
            'calling_p':line[:2], 
            'call':{player.split(":")[0]:set(player.split(":")[1][1:-1].split(',')) for player in line[3:].split()[:-1]},
            'status':int(line[-2])
        }
    
    def parse_ask(self, line):
        return {
            'asking_p':line[:2],
            'asked_p':line[3:5],
            'card':line[6:8],
            'status':int(line[-2])
        }

    def update_score(self, team_index):
        self.score[team_index] += 1

    def try_call(self, call, prev_hands, i):
        new_hands = copy.deepcopy(prev_hands)
        calling_p, call, status = call.values()
        p_index = (self.initials_to_index(calling_p) + status + 1) % 2
        self.update_score(p_index)

        try:
            for player, cards in call.items():
                for card in cards:
                    new_hands[player].remove(card)
            if not status:
                raise ParseError(f"Invalid unsuccessful set call on line {i+2} {prev_hands}")
        except KeyError as e:
            if status:
                raise ParseError(f"Invalid successful set call on line {i+2} {prev_hands}") from e
            called_cards = set()
            for v in call.values():
                called_cards = called_cards.union(v)
            for hand in new_hands.values():
                hand -= called_cards
        return new_hands
    
    def try_ask(self, ask, prev_hands, i):
        new_hands = copy.deepcopy(prev_hands)
        try:
            new_hands[ask['asked_p']].remove(ask['card'])
            new_hands[ask['asking_p']].add(ask['card'])
            return new_hands
        except KeyError as e:
            raise ParseError(f"Invalid transaction on line {i+2}") from e
        
    def encode_player(self, initials):
        p = np.zeros(8, dtype=int)
        p[self.initials_to_index(initials)] = 1
        return p
    
    def construct_call_vector(self, call):
        calling_p, call, status = call.values()
        self.rewards.append(-1 if self.initials_to_index(calling_p) == status else 1) # award for beneficiary
        state_array = np.array([1]) # call indicator

        # encoding caller
        state_array = np.concatenate((state_array, self.encode_player(calling_p)))

        # encoding calls
        called_set = False
        v_call = np.array([])
        for player in self.teammates(calling_p):
            c = np.zeros(6)
            if player in call:
                for card in call[player]:
                    if not called_set:
                        called_set = True
                        state_array = np.concatenate((state_array, card_to_vector[card][6:]))
                    c += card_to_vector[card][:6]
            c = c > 0
            v_call = np.concatenate((v_call, c))

        # encoding status
        state_array = np.concatenate((state_array, v_call.T, np.zeros(24-len(v_call.T)), np.array([status]), np.zeros(11)))
        return state_array

    def construct_ask_vector(self, ask):
        self.rewards.append(0)
        asking_p, asked_p, card, status = ask.values()
        return np.concatenate((np.array([0]), 
                               self.encode_player(asking_p),
                               self.encode_player(asked_p),
                               card_to_vector[card],
                               np.array([status]),
                               np.zeros(21)))
        
    def to_state(self):
        self.state = np.zeros((len(self.datarows[1:-1]),54), dtype=int)
        for i, line in enumerate(self.datarows[1:-1]):
            self.state[i] = self.construct_call_vector(self.parse_call(line)) if ":" in line else self.construct_ask_vector(self.parse_ask(line))

    def encode_hand(self, hand, flatten=True):
        hand_vector = np.zeros((9,6), dtype=int)
        for card in hand:
            hand_vector[np.where(sets == card)[0][0]][np.where(sets == card)[1][0]] = 1
        return hand_vector.flatten() if flatten else hand_vector
    
    def get_state(self, i, player, pad=True):
        i = i if i < len(self.hands) else -1
        return np.concatenate((self.encode_hand(self.hands[i][player]).reshape(1,54), 
                                self.state[:i][::-1],
                                np.zeros((200-i-1,54))))

    def memory(self, player):
        is_ask = lambda i: not self.state[i][0] and all(self.state[i][1:9] == self.encode_player(player))
        is_call = lambda i: self.state[i][0] and all(self.state[i][1:9] == self.encode_player(player))
        return [{
            'state': self.get_state(i, player), # invert sequential order, pad up to 200,
            'reward': self.rewards[i],
            'action': {
                'call': is_call(i),
                'call_set': self.state[i][1:1+9] if is_call(i) else None,
                'call_cards': self.state[i][1+9:1+9+24] if is_call(i) else None,
                'ask_person': self.state[i][9:9+8] if is_ask(i) else None, 
                'ask_set': self.state[i][9+8:9+8+9] if is_ask(i) else None,
                'ask_card': self.state[i][9+8+9:9+8+9+6] if is_ask(i) else None,
            },
            'next_state': self.get_state(i+1, player),
            'mask_dep': self.mask_dep(i, player),
            'next_mask_dep': self.mask_dep(i+1, player)
        } for i in range(len(self.hands)-1)]
    
    def sets_remaining(self, i):
        cards_remaining = np.zeros((9,6), dtype=int)
        for hand in self.hands[i].values():
            cards_remaining += self.encode_hand(hand, flatten=False)
        return (np.sum(cards_remaining, axis=1) > 0).astype(int)
    
    def mask_dep(self, i, player):
        return {
            'agent_index': self.players.index(player),
            'hand': self.encode_hand(self.hands[i][player], flatten=False), # 9x6
            'sets_remaining': self.sets_remaining(i),
            'cards_remaining': np.array([len(hand) for hand in self.hands[i].values()])
        }

    def verify(self):
        hands = [self.init_hands]
        score = [int(self.datarows[-1].split(" ")[0]), int(self.datarows[-1].split(" ")[1])]
        for i, line in enumerate(self.datarows[1:-1]):
            if ":" in line: # calling
                hands.append(self.try_call(self.parse_call(line), hands[-1], i))
            elif self.parse_ask(line)['status']: # asking
                ask = self.parse_ask(line)
                hands.append(self.try_ask(ask, hands[-1], i))
        if score != self.score:
            raise ParseError(f"Score is invalid {score} {self.score}")
        if not any(hands[-1].values()):
            self.hands = hands