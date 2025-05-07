from agent import QLearningAgent
import pickle

with open('memories_extended.pkl', 'rb') as f:
    memories = pickle.load(f)

agent = QLearningAgent(memories)
agent.load_model('models/fish_agent.pth')
agent.train_self_play(300, hand_epochs=5, q_epochs=5, path='models/fish_agent.pth')