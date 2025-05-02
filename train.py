from agent import QLearningAgent

agent = QLearningAgent()
agent.load_model('models/fish_agent.pth')
agent.train_self_play(2000, hand_epochs=5, q_epochs=20, path='models/fish_agent.pth')