from agent import QLearningAgent
from verify_data import FishGame, ParseError
import os
import pickle
import torch
import multiprocessing as mp

def process_file(filename, data_dir='data'):
    filepath = os.path.join(data_dir, filename)
    file_memories = []
    try:
        with open(filepath, 'r') as f:
            print(f"{filename}")
            game = FishGame(f.readlines())
            for player in game.players:
                for _ in range(100):
                    game.shuffle()
                    file_memories.append(game.memory(player))
        return file_memories
    except ParseError as e:
        print(f"{filename}: {e}")
        return []

if os.path.isfile('memories_extended.pkl'):
    with open('memories_extended.pkl', 'rb') as f:
        memories = pickle.load(f)
else:
    memories = []
    filenames = os.listdir('data')
    
    if torch.cuda.is_available():
        num_processes = min(mp.cpu_count(), 8)
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(process_file, filenames)
        for result in results:
            memories.extend(result)
    else:
        for filename in filenames:
            filepath = os.path.join('data', filename)
            with open(filepath, 'r') as f:
                try:
                    print(f"{filename}")
                    game = FishGame(f.readlines())
                    for player in game.players:
                        for _ in range(100):
                            game.shuffle()
                            memories.append(game.memory(player))
                except ParseError as e:
                    print(f"{filename}: {e}")
                    break
    with open('memories_extended.pkl', 'wb') as f:
        pickle.dump(memories, f)
        
agent = QLearningAgent(memories) 
agent.load_model('models/fish_agent.pth')
agent.train_self_play(2000, update_rate=1, hand_epochs=10, q_epochs=5, path='models/fish_agent.pth')