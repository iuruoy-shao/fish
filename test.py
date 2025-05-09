from verify_data import SimulatedFishGame
from agent import QLearningAgent
import numpy as np
import copy

agent = QLearningAgent()
agent.load_model('models/fish_agent.pth')

np.set_printoptions(suppress=True)
player = "YS"
with open("data/test.txt", "r") as f:
    lines = f.readlines()
game = SimulatedFishGame(8)
game.datarows = lines
game.init_hands = {player.split(":")[0]:set(player.split(":")[1][1:-1].split(',')) 
                           for player in game.datarows[0].split()}
game.hands = [game.init_hands]
game.players = list(game.init_hands.keys())
game.verify()

game.rotate(player)
game_state = game.to_state()
state = agent.tensor(np.stack([game.get_state(i, game_state) for i in range(len(game.hands))]))
mask = agent.action_masks(*agent.unpack_memory([game.mask_dep(len(game.hands)-1, player)]).values())
pred_hands, action = agent.act(state, mask)
new_hands = copy.deepcopy(game.hands[-1])
if action['call'][0] > action['call'][1]:
    print(game.handle_call(action, new_hands, player))
else:
    print(game.handle_ask(action, new_hands, player, help_call=False))