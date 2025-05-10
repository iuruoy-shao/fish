from project.agent import QLearningAgent
import pickle

# with open('memories_extended.pkl', 'rb') as f:
#     memories = pickle.load(f)
# with open('stored_memories.pkl', 'rb') as f:
#     stored_memories = pickle.load(f)

# print(f"memories: {len(memories)}, stored_memories: {len(stored_memories)}")

agent = QLearningAgent()
agent.load_model('../models/model.pth')
agent.train_self_play(75,5,5,'../models/model.pth')