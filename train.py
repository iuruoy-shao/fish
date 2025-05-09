from agent import QLearningAgent
import pickle

with open('memories_extended.pkl', 'rb') as f:
    memories = pickle.load(f)
with open('stored_memories.pkl', 'rb') as f:
    stored_memories = pickle.load(f)

print(f"memories: {len(memories)}, stored_memories: {len(stored_memories)}")

agent = QLearningAgent()
agent.train_on_data(stored_memories, 20, 0)
agent.save_model('models/fish_agent3.pth')
agent.train_on_data(memories, 0, 25)
agent.save_model('models/fish_agent3.pth')