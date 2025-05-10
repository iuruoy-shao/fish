from project.agent import QLearningAgent
import pickle

# with open('memories_extended.pkl', 'rb') as f:
#     memories = pickle.load(f)
with open('project/train/stored_memories_2.pkl', 'rb') as f:
    stored_memories = pickle.load(f)

# print(f"memories: {len(memories)}, stored_memories: {len(stored_memories)}")

agent = QLearningAgent()
agent.load_model('project/models/model.pth')
# agent.train_self_play(75,5,5,'project/models/model.pth')
agent.train_on_data(stored_memories, 20, 20)
agent.save_model('project/models/model.pth')