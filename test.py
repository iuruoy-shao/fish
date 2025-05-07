from verify_data import FishGame
from agent import QLearningAgent
import numpy as np

player = "P1"

with open("data/test.txt", "r") as f:
    lines = f.readlines()
game = FishGame(lines)

agent = QLearningAgent()
agent.load_model('models/fish_agent.pth')

game.rotate(player)
game_state = game.to_state()
state = agent.tensor(np.stack([game.get_state(i, game_state) for i in range(len(game.hands))]))
mask = agent.action_masks(*agent.unpack_memory([game.mask_dep(len(game.hands)-1, player)]).values())
pred_hands, action = agent.act(state, mask)
print(action)