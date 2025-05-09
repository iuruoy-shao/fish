from agent import QLearningAgent
import pickle

# with open('memories_extended.pkl', 'rb') as f:
#     memories = pickle.load(f)

agent = QLearningAgent()
agent.load_model('models/fish_agent.pth')
# agent.train_on_data(memories[::25], hand_epochs=0, q_epochs=10)
agent.train_self_play(100, hand_epochs=5, q_epochs=5, path='models/fish_agent.pth')