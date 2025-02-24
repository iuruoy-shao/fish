# verifies that the data steps are correct
import copy

class FishGame:
    def __init__(self, datarows):
        # load initial cards
        init_hands = {player.split(":")[0]:set(player.split(":")[1][1:-1].split(',')) for player in datarows[0].split()}
        self.hands = [init_hands]
        for i, line in enumerate(datarows[1:-1]):
            if ":" in line: # calling
                status = line[-2]
                new_hands = copy.deepcopy(self.hands[-1])
                call = {player.split(":")[0]:set(player.split(":")[1][1:-1].split(',')) for player in line[3:].split()[:-1]}
                if status == "1":
                    try:
                        for player, cards in call.items():
                            for card in cards:
                                new_hands[player].remove(card)
                        self.hands.append(new_hands)
                    except KeyError:
                        print(f"Invalid successful set call on line {i+2} {self.hands[-1]}")
                        break
                else:
                    try:
                        for player, cards in call.items():
                            for card in cards:
                                new_hands[player].remove(card)
                        print(f"Invalid unsuccessful set call on line {i+2} {self.hands[-1]}")
                        break
                    except KeyError:
                        called_cards = set()
                        for v in call.values():
                            called_cards = called_cards.union(v)
                        new_hands = copy.deepcopy(self.hands[-1])
                        for player in new_hands.keys():
                            new_hands[player] -= called_cards
                        self.hands.append(new_hands)
            elif line[-2] == "1":
                asked_player = line[3:5]
                card = line[6:8]

                new_hands = copy.deepcopy(self.hands[-1])
                asking_player = line[:2]
                try:
                    new_hands[asked_player].remove(card)
                    new_hands[asking_player].add(card)
                except KeyError:
                    print(f"Invalid transaction on line {i+2}")
                    break
                self.hands.append(new_hands)
        print("data is valid, score needs to be manually verified")
if __name__ == "__main__":
    with open("12-3_2:27.txt", "r") as f:
        game = FishGame(f.readlines())