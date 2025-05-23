import copy
import json
import numpy as np
import random

agent_initials = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8']
CALL_LEN = 42
ASK_LEN = 32

rewards = {
    'correct_call': 1,
    'incorrect_call': -1,
    'correct_team_call': 1,
    'incorrect_team_call': -1,
    'correct_opponent_call': -1,
    'incorrect_opponent_call': 0,
    'correct_ask': .1,
    'incorrect_ask': -.1,
    'correct_team_ask': .1,
    'incorrect_team_ask': -.1,
    'correct_opponent_ask': -.1,
    'incorrect_opponent_ask': 0,
}

with open('project/sets.json','r') as f:
    global sets_array
    global sets
    global card_to_vector
    global all_cards
    sets = json.load(f)
    sets_array = np.array(sets)
    card_to_vector = {}
    for card in sets_array.flatten():
        vector = np.zeros(15, int)
        vector[np.where(sets_array == card)[0][0]] = 1
        vector[np.where(sets_array == card)[1][0] + 9] = 1
        card_to_vector[str(card)] = vector
    all_cards = list(card_to_vector.keys())

class ParseError(Exception):
    pass

class FishGame:
    def __init__(self, datarows):
        self.datarows = datarows
        self.init_hands = {player.split(":")[0]:set(player.split(":")[1][1:-1].split(',')) 
                           for player in self.datarows[0].split()}
        self.players = list(self.init_hands.keys())
        self.score = [0, 0]
        self.last_indices = {}
        self.rewards = [] # encodes for even team, reverse for odd team
        self.verify()
        self.all_pred_hands = []

    def shuffle(self):
        global sets_array
        global card_to_vector
        global sets
        global all_cards
        np.random.shuffle(sets_array) # shuffle rows
        np.apply_along_axis(np.random.shuffle, 1, sets_array) # shuffle within row
        sets = sets_array.tolist()
        card_to_vector = {}
        for card in sets_array.flatten():
            vector = np.zeros(15, int)
            vector[np.where(sets_array == card)[0][0]] = 1
            vector[np.where(sets_array == card)[1][0] + 9] = 1
            card_to_vector[str(card)] = vector
        all_cards = list(card_to_vector.keys())
        team0 = self.players[::2]
        team1 = self.players[1::2]
        random.shuffle(team0)
        random.shuffle(team1)
        self.players = [team1[i//2] if i % 2 else team0[i//2] for i in range(len(self.players))]

    def initials_to_index(self, initials):
        return self.players.index(initials)

    def teammates(self, initials):
        return self.players[self.initials_to_index(initials) % 2::2]
    
    def set_index(self, card):
       return np.where(sets_array == card)[0][0]
    
    def card_index(self, card):
        return np.where(sets_array == card)[1][0]

    def vector_to_card(self, card_vector):
        set_index = np.argmax(card_vector[:9])
        card_index = np.argmax(card_vector[9:])
        return sets_array[set_index][card_index]

    def flatten_card_vector(self, card_vector):
        set_index = np.argmax(card_vector[:9])
        card_index = np.argmax(card_vector[9:])
        vector = np.zeros(54, dtype=int)
        vector[set_index*6 + card_index] = 1
        return vector

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
        if self.initials_to_index(ask['asking_p']) % 2 == self.initials_to_index(ask['asked_p']) % 2:
            raise ParseError(f"Invalid ask on line {i+2}, same team")
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

        is_self = self.initials_to_index(calling_p) == 0
        same_team = not self.initials_to_index(calling_p) % 2
        self.rewards.append(
            (rewards['correct_call'] if is_self
             else rewards['correct_team_call'] if same_team
             else rewards['correct_opponent_call'])
            if status else 
            (rewards['incorrect_call'] if is_self
             else rewards['incorrect_team_call'] if same_team 
             else rewards['incorrect_opponent_call'])
        )

        state_array = self.encode_player(calling_p)

        called_set = False
        v_call = np.array([])
        for player in self.teammates(calling_p):
            c = np.zeros(6)
            if player in call:
                for card in call[player]:
                    if not called_set:
                        called_set = True
                        state_array = np.concatenate((state_array, card_to_vector[card][:9]))
                    c += card_to_vector[card][9:]
            c = c > 0
            v_call = np.concatenate((v_call, c))

        state_array = np.concatenate((state_array, 
                                      np.pad(v_call,(0,24-len(v_call))).reshape(4,6).T.flatten(), 
                                      np.array([status])))
        return state_array

    def construct_ask_vector(self, ask):
        asking_p, asked_p, card, status = ask.values()

        is_self = self.initials_to_index(asking_p) == 0
        same_team = not self.initials_to_index(asking_p) % 2
        self.rewards.append(
            (rewards['correct_ask'] if is_self 
             else rewards['correct_team_ask'] if same_team
             else rewards['correct_opponent_ask'])
            if status else 
            (rewards['incorrect_ask'] if is_self 
             else rewards['incorrect_team_ask'] if same_team
             else rewards['incorrect_opponent_ask'])
        )
        
        return np.concatenate((self.encode_player(asking_p),
                               self.encode_player(asked_p),
                               card_to_vector[card],
                               np.array([status])))
        
    def to_state(self):
        return [
            np.concatenate((
                (self.construct_call_vector(self.parse_call(line)), np.zeros(ASK_LEN, dtype=int))
                if ":" in line else 
                (np.zeros(CALL_LEN, dtype=int), self.construct_ask_vector(self.parse_ask(line)))
            ))
            for line in self.datarows[1:-1]
        ]

    def encode_hand(self, hand, flatten=True):
        hand_vector = np.zeros((9,6), dtype=int)
        for card in hand:
            hand_vector[np.where(sets_array == card)[0][0]][np.where(sets_array == card)[1][0]] = 1
        return hand_vector.flatten() if flatten else hand_vector
    
    def encode_all_hands(self, i, player=None):
        return np.stack([self.encode_hand(self.hands[i][p]) for p in self.players] 
                        + [self.encode_hand({})] * (8-len(self.players)), axis=0)
    
    def decode_pred_hands(self, pred_hands): # shape, 8 x 54
        return {all_cards[i]:{self.players[j]:pred_hands[j,i] 
                              for j in range(len(self.players))} 
                for i in range(54)}
    
    def encode_pred_hands(self, pred_hands): # shape, 8 x 54
        array = np.zeros((8,54))
        for i, card in enumerate(all_cards):
            for j, player in enumerate(self.players):
                array[j,i] = pred_hands[card][player]
        return array
                
    def get_state(self, i, ordered_state):
        i = i if i < len(self.hands) else -1
        return np.concatenate((self.cards_remaining(i), ordered_state[i-1] if i else np.zeros(CALL_LEN+ASK_LEN, dtype=int)))

    def rotate(self, player):
        self.rewards = []
        self.players = self.players[self.initials_to_index(player):] + self.players[:self.initials_to_index(player)]
    
    def last_hand_index(self, player):
        if player not in self.last_indices:
            for i, hand in enumerate(self.hands[::-1]):
                if len(hand[player]):
                    self.last_indices[player] = len(self.hands)-i
                    break
        return self.last_indices[player]

    def memory_item(self, i, player, state, hand_encodings, last):
        def ask_vector(r, c):
            a = np.zeros((4, 54), dtype=int)
            a[r][c] = 1
            return a.flatten()
        is_ask = lambda i: not any(state[i][:CALL_LEN]) and (state[i][CALL_LEN:CALL_LEN+8] == self.encode_player(player)).all()
        is_call = lambda i: any(state[i][:CALL_LEN]) and (state[i][:8] == self.encode_player(player)).all()
        
        return {
            'state': self.get_state(i, state),
            'hands': hand_encodings[i],
            'next_hands': hand_encodings[i+1] if i+1 < last else np.zeros((8, 54), dtype=int),
            **({
                'predicted_hands': self.encode_pred_hands(self.all_pred_hands[i][player]),
                'next_predicted_hands': self.encode_pred_hands(self.all_pred_hands[i+1][player]) 
                                        if i+1 < last 
                                        else np.zeros((8, 54), dtype=int)
                } if self.all_pred_hands else {}),
            'reward': np.array(self.rewards[i]).reshape(-1),
            'action': {
                'call': np.array([1,0] if is_call(i) else [0,1]),
                'call_set': state[i][8:8+9] if is_call(i) else None,
                'call_cards': state[i][8+9:8+9+24].reshape((6,4)) if is_call(i) else None,
                'ask': ask_vector(np.argmax(state[i][CALL_LEN+8:CALL_LEN+8+8][1::2]),
                                  np.argmax(state[i][CALL_LEN+8+8:CALL_LEN+8+8+9]) * 6 
                                  + np.argmax(state[i][CALL_LEN+8+8+9:CALL_LEN+8+8+9+6])) if is_ask(i) else None,
            },
            'mask_dep': self.mask_dep(i, player),
            'next_mask_dep': self.mask_dep(i+1, player)
        }

    def memory(self, player, pick_last=False, return_sep=False):
        self.rotate(player)
        state = self.to_state()
        hand_encodings = [self.encode_all_hands(i) for i in range(len(state)+1)]

        is_call = lambda i: any(state[i][:CALL_LEN]) and (state[i][:8] == self.encode_player(player)).all()
        last = self.last_hand_index(player)

        memory = []
        call_memory = []
        ask_memory = []
        for i in ([last-3] if pick_last else range(last)):
            entry = self.memory_item(i, player, state, hand_encodings, last)
            memory.append(entry)
            if return_sep:
                if is_call(i):
                    call_memory.append(entry)
                else:
                    ask_memory.append(entry)
        return (memory, ask_memory, call_memory) if return_sep else memory
    
    def sets_remaining(self, i):
        cards_remaining = np.zeros((9,6), dtype=int)
        for hand in self.hands[i].values():
            cards_remaining += self.encode_hand(hand, flatten=False)
        return (np.sum(cards_remaining, axis=1) > 0).astype(int)
    
    def cards_remaining(self, i):
        return np.array([len(self.hands[i][player]) for player in self.players] 
                        + ([0, 0] if len(self.players) == 6 else []))
    
    def mask_dep(self, i, player):
        return {
            'hand': self.encode_hand(self.hands[i][player], flatten=False), # 9x6
            'sets_remaining': self.sets_remaining(i),
            'cards_remaining': self.cards_remaining(i),
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
            else: # unsuccessful ask
                hands.append(hands[-1])
        if score != self.score:
            raise ParseError(f"Score is invalid {score} {self.score}")
        if any(hands[-1].values()):
            print(f"Game incomplete {hands[-1]}")
        self.hands = hands

class SimulatedFishGame(FishGame):
    def __init__(self, n_players):
        self.help_threshold = 0.5
        self.init_hands = {}
        self.n_players = n_players
        self.players = agent_initials[:n_players]
        self.assign_hands()
        self.hands = [self.init_hands]
        self.rewards = []
        self.last_indices = {}
        self.datarows = [f'{" ".join([f"{player}:{{{",".join(self.init_hands[player])}}}" 
                         for player in self.players])}\n', 'temp_score']
        self.turn = random.choice(self.players)
        self.score = [0, 0]

    def ended(self):
        return not any(self.hands[-1].values())
    
    def asking_ended(self):
        return not any(self.hands[-1][player] for player in self.players[1::2]) or not any(self.hands[-1][player] for player in self.players[::2])

    def assign_hands(self):
        cards = list(card_to_vector.keys())[:-6 if self.n_players == 8 else None] # remove extra set 
        hand_length = len(cards) // self.n_players
        random.shuffle(cards)
        for i, initials in enumerate(self.players):
            self.init_hands[initials] = set(cards[i*hand_length:(i+1)*hand_length])

    def random_pass(self):
        if valid_teammates := [
            teammate
            for teammate in self.teammates(self.turn)
            if self.hands[-1][teammate]
        ]:
            self.turn = random.choice(valid_teammates)

    def players_with_cards(self):
        return [player for player in self.players if self.hands[-1][player]]
    
    def parse_action(self, action, player):
        new_hands = copy.deepcopy(self.hands[-1])
        self.rotate(player)
        if int(player[-1]) % 2 and (monopoly := self.monopolized_set(new_hands)) and len(self.datarows) > 10:
            move = self.handle_call(None, new_hands, player, force_call=monopoly)
        else:
            is_call = action['call'][0] > action['call'][1]
            if is_call:
                move = self.handle_call(action, new_hands, player)
            else:
                move = self.handle_ask(action, new_hands, player)

        self.hands.append(new_hands)
        self.datarows.insert(-1, move)

        if not new_hands[self.turn]: # if the player whose turn it is runs out of cards, pass to teammate with cards
            self.random_pass()
            
    def monopolized_set(self, hands):
        team0_hands = set().union(*(hands[ref_player] for ref_player in self.players[::2]))
        for i, card_set in enumerate(sets):
            card_set = set(card_set)
            if card_set.issubset(team0_hands):
                return i, {ref_player:card_set.intersection(hands[ref_player]) for ref_player in self.players[::2]}
        return False

    def handle_call(self, action, new_hands, player, force_call=None):
        success = True
        if force_call:
            call_set, card_assignments = force_call
            print(f"helping call... {card_assignments}")
        else:
            call_set = np.argmax(action['call_set'])
            call_cards = np.argmax(action['call_cards'], axis=1)

            card_assignments = {ref_player:set() for ref_player in self.players[::2]}
            for card_index, player_index in enumerate(call_cards):
                ref_player = self.players[::2][player_index]
                card = sets[call_set][card_index]
                success = success and (card in new_hands[ref_player])
                card_assignments[ref_player].add(card)

        for hand in new_hands.values():
            hand -= set(sets[call_set])
        return f'{player} {" ".join([f"{ref_player}:{{{",".join(card_assignments[ref_player])}}}" 
                           for ref_player in self.players[::2] if card_assignments[ref_player]])} {int(success)}\n'

    def handle_ask(self, action, new_hands, player, help_call=True):
        ask = action['ask'].reshape(4, 54)
        ask_person, ask_card = np.unravel_index(np.argmax(ask, axis=None), ask.shape)
        ask_person = self.players[1::2][ask_person]
        card = all_cards[ask_card]
        success = card in new_hands[ask_person]
        if help_call and int(player[-1]) % 2 and not success and random.random() < self.help_threshold:
            if helped_card := self.pick_successful_card(player, ask_person, new_hands, card):
                card = helped_card
                success = True
                print(f"helping ask... {card}")
        if success:
            new_hands[ask_person].remove(card)
            new_hands[player].add(card)
        else:
            self.turn = ask_person
        return f"{player} {ask_person} {card} {int(success)}\n"
    
    def pick_successful_card(self, asking_player, ask_person, new_hands, card):
        og_set_index = self.set_index(card)
        their_hand = list(new_hands[ask_person])
        their_hand_index = [self.set_index(card) for card in their_hand]
        my_hand_index = [self.set_index(card) for card in new_hands[asking_player]]
        if og_set_index in their_hand_index:
            return their_hand[their_hand_index.index(og_set_index)]
        shared_cards = set(their_hand_index).intersection(set(my_hand_index))
        if shared_cards:
            return their_hand[their_hand_index.index(list(shared_cards)[0])]
        return False