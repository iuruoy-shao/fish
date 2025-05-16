from project.agent import QLearningAgent
import pickle

# with open('project/train/memories_extended.pkl', 'rb') as f:
#     memories = pickle.load(f)

# print(f"stored_memories: {len(stored_memories)} call_memories: {len(call_memories)} ask_memories: {len(ask_memories)}")

agent = QLearningAgent()
agent.epsilon = 0 # no exploring lol
agent.load_model('project/models/model.pth')
# agent.train_on_data(memories, 5, 15)
# agent.save_model('project/models/model.pth')
# agent.train_self_play(50, 5, 20, 'project/models/model.pth')

# with open('project/train/stored_memories_2.pkl', 'rb') as f:
#     stored_memories = pickle.load(f)
# with open('project/train/call_memories_2.pkl', 'rb') as f:
#     call_memories = pickle.load(f)
with open('project/train/ask_memories_2.pkl', 'rb') as f:
    ask_memories = pickle.load(f)
# print(f"stored_memories: {len(stored_memories)} call_memories: {len(call_memories)} ask_memories: {len(ask_memories)}")
agent.train_on_data(ask_memories, 20, 0, reset_lr=True)
agent.save_model('project/models/model.pth')
# agent.train_on_data(call_memories, 100, 0, reset_lr=True)
# agent.save_model('project/models/model.pth')
# agent.train_on_data(stored_memories, 5, 15, reset_lr=True)
# agent.save_model('project/models/model.pth')