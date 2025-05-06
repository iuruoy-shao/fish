from verify_data import SimulatedFishGame, CALL_LEN, ASK_LEN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
import pickle
from tqdm import tqdm
import itertools
import time
import os

class HandPrediction(nn.Module):
    def __init__(self):
        super(HandPrediction, self).__init__()
        self.rnn = nn.LSTM(8+CALL_LEN+ASK_LEN, 128, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 8*54)
        
    def forward(self, x, mask):
        out, _ = self.rnn(x)
        return F.softmax(F.relu(self.fc(self.dropout(out)))
                         .reshape(-1,8,54)
                         .masked_fill(~mask, 0), dim=1).masked_fill(~mask, 0)

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=6, stride=6) # Output: (batch, 16, 9 sets)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1) # Output: (batch, 32, 9 sets)
        self.pool = nn.AdaptiveAvgPool1d(1) # Output: (batch, 32, 1)

        self.fc1 = nn.Linear(32, 32)
        self.dropout = nn.Dropout(0.5)

        self.to_call = nn.Linear(32, 2)
        self.pick_call_set = nn.Linear(32, 9)
        self.pick_call_cards = nn.Linear(32 + 9, 24)

        self.pick_person = nn.Linear(32, 4)
        self.pick_ask_set = nn.Linear(32 + 4, 9)
        self.pick_ask_card = nn.Linear(32 + 4 + 9, 6)

    def forward(self, x, action_masks):
        x = x.reshape(-1, 8, 54)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(self.pool(x), 1)
        x = self.dropout(F.relu(self.fc1(x)))

        to_call = self.to_call(x)
        call_set = self.pick_call_set(x)
        call_cards = torch.reshape(self.pick_call_cards(torch.cat((x, call_set), dim=1)), (-1, 6, 4))

        ask_person = self.pick_person(x)
        ask_set = self.pick_ask_set(torch.cat((x, ask_person), dim=1))
        ask_card = self.pick_ask_card(torch.cat((x, ask_person, ask_set), dim=1))

        return {
            'call': to_call,
            'call_set': call_set.masked_fill(~action_masks['call_set'], -1e9),
            'call_cards': call_cards.masked_fill(~action_masks['call_cards'], -1e9),
            'ask_person': ask_person.masked_fill(~action_masks['ask_person'], -1e9), #TODO: combine this to a 4x54 instead for easier relations
            'ask_set': ask_set.masked_fill(~action_masks['ask_set'], -1e9),
            'ask_card': ask_card
        }

class QLearningAgent:
    def __init__(self, real_data=None):
        if real_data is None:
            real_data = []
        self.device = torch.device("mps" if torch.backends.mps.is_available() 
                                   else "cuda" if torch.cuda.is_available() 
                                   else "cpu")

        self.hand_predictor = HandPrediction().to(self.device)
        self.hand_optimizer = torch.optim.Adam(self.hand_predictor.parameters(), lr=0.001)
        self.hand_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.hand_optimizer, mode='min', factor=0.8, patience=10, min_lr=1e-6
        )

        self.q_network = QNetwork().to(self.device)
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.q_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.q_optimizer, mode='min', factor=0.8, patience=5, min_lr=1e-6
        )
        self.loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.real_data = real_data

        self.gamma = 0.99    # discount factor
        self.epsilon = 0.1   # exploration rate
        self.hand_batch_size = 3
        self.q_batch_size = 1024

    def tensor(self, x, as_bool=False):
        if as_bool:
            return torch.BoolTensor(np.array(x)).to(self.device)
        return torch.FloatTensor(np.array(x)).to(self.device)

    def max_q(self, action, i):
        if all(action['ask_set'][i] < -9e8):
            return torch.tensor(0).to(self.device)
        if torch.argmax(action['call'][i]) == 0:
            return torch.max(action['call_cards'][i])
        return torch.max(action['ask_card'][i])
    
    def target_q(self, max_q_next, reward):
        return reward + self.gamma * max_q_next
    
    def current_q(self, agent_actions, player_actions):
        return torch.sum(agent_actions * player_actions, dim=0)

    def q_loss(self, action, next_action, player_action, reward):
        agent_actions = []
        player_actions = []
        rewards = []
        next_qs = []

        for i in range(len(reward)):
            this_reward = reward[i][0]
            this_next_q = self.max_q(next_action, i)
            for act in player_action.keys():
                if player_action[act][i] is None:
                    continue # skip Nones
                this_action = action[act][i]
                this_player_action = self.tensor(player_action[act][i])
                if act == 'call_cards':
                    agent_actions.extend(this_action)
                    player_actions.extend(this_player_action)
                    rewards += [this_reward] * 6
                    next_qs += [this_next_q] * 6
                else:
                    agent_actions.append(this_action)
                    player_actions.append(this_player_action)
                    rewards.append(this_reward)
                    next_qs.append(this_next_q)
        return self.loss(self.current_q(pad_sequence(agent_actions), pad_sequence(player_actions)), 
                         self.target_q(torch.stack(next_qs), self.tensor(rewards)))
    
    def accuracy(self, pred_hands, episode):
        pred_hands = torch.concat(pred_hands).cpu().detach().numpy()
        choices = np.argmax(pred_hands, axis=1)
        one_hot = np.zeros_like(pred_hands)
        one_hot[np.arange(pred_hands.shape[0])[:,None], choices, np.arange(54)[None,:]] = 1
        cards_remaining = np.sum(np.stack(episode['mask_dep']['sets_remaining']), axis=1) * 6
        guarantee = np.stack(episode['mask_dep']['cards_remaining'])[:,0]
        accuracies = ((one_hot * episode['hands']).sum((1,2)) - guarantee) / (cards_remaining - guarantee)
        return accuracies[~np.isnan(accuracies)]
    
    def action_masks(self, hand, sets_remaining, cards_remaining):
        cards_remaining = np.array(cards_remaining)
        return {
            'hands': self.tensor(np.tile((cards_remaining > 0).reshape((-1,8,1)), (1,1,54)) 
                                 * np.tile(np.repeat(sets_remaining, 6, axis=1)[:,np.newaxis,:], (1,8,1))
                                 * np.concat((np.reshape(hand, (-1,1,54)), 
                                              ~(np.tile(np.reshape(hand, (-1,1,54)), (1,7,1)) > 0)), axis=1), 
                                 as_bool=True), # cards & players still in game
            'call_set': self.tensor(sets_remaining, as_bool=True),  # the sets that remain
            'call_cards': self.tensor(np.tile((cards_remaining[:,::2] > 0)[:,np.newaxis,:], (1,6,1)), as_bool=True),  # the players on the team that still have cards
            'ask_person': self.tensor(cards_remaining[:,1::2] > 0, as_bool=True),  # the players on the opposing team that still have cards
            'ask_set': self.tensor(np.sum(hand, axis=2) > 0, as_bool=True)  # the sets that the player holds
        }
    
    def unpack_memory(self, batch):
        return {
            key: (
                self.unpack_memory([item[key] for item in batch])
                if isinstance(batch[0][key], dict)
                else [item[key] for item in batch]
            )
            for key in batch[0].keys()
        }
        
    def shuffle_item(self, item, idx_list):
        if isinstance(item, dict):
            return {k: self.shuffle_item(v, idx_list) for k, v in item.items()}
        elif isinstance(item, torch.Tensor):
            return item[idx_list]
        else:
            return [item[i] for i in idx_list]

    def shuffle_memory(self):
        indices = np.arange(0,len(self.memory['state']))
        random.shuffle(indices)
        return {key: self.shuffle_item(value, indices) for key, value in self.memory.items()}
    
    def shuffle_episodes(self, episode_indices, memory):
        keys =  list(episode_indices.keys())
        random.shuffle(keys)
        episode_indices = {key: episode_indices[key] for key in keys}
        
        indices = np.concatenate(list(episode_indices.values()))
        index = 0
        for i, episode in episode_indices.items():
            episode_indices[i] = np.arange(index, index + len(episode))
            index += len(episode)
        return episode_indices, {key: self.shuffle_item(value, indices) for key, value in memory.items()}
    
    def pick_batch(self, memory, indices):
        start, end = indices
        return {
            key: self.pick_batch(memory[key], (start, end))
            if isinstance(memory[key], dict)
            else memory[key][start:end]
            for key in memory.keys()
        }
    
    def handle_q_batch(self, batch, pred_hands=True):
        current_q = self.q_network(self.tensor(batch['predicted_hands' if pred_hands else 'hands']), batch['action_masks'])
        next_q = self.q_network(self.tensor(batch['next_predicted_hands' if pred_hands else 'next_hands']), batch['next_action_masks'])
        return self.q_loss(current_q, next_q, batch['action'], batch['reward'])
    
    def handle_hand_batch(self, batch, episode_lengths):
        reshape = lambda x: torch.reshape(torch.swapaxes(x[:,1:],1,2), (-1,7))
        i = 0
        pred_hands = []
        for episode_length in episode_lengths:
            episode = self.pick_batch(batch, (i,i+episode_length))
            i += episode_length
            pred_hands.append(self.hand_predictor(self.tensor(episode['state']),
                                                  episode['action_masks']['hands']))
        stacked_pred_hands = reshape(torch.concat(pred_hands))
        accuracy = np.average(self.accuracy(pred_hands, batch).tolist())
        batch_loss = self.cross_entropy_loss(stacked_pred_hands, reshape(self.tensor(batch['hands'])))
        return batch_loss, accuracy
    
    def train_q_network(self, n_epochs, lr_schedule=True, use_tqdm=True):
        pred_hands = 'predicted_hands' in self.memory
        train_size = int(0.8 * len(self.memory['state']))
        test_batch = self.pick_batch(self.memory, (train_size, len(self.memory['state'])))
        self.memory = self.pick_batch(self.memory, (0, train_size))
        
        if use_tqdm:
            t = tqdm(range(n_epochs), desc="Training Q-Network")
        for epoch in t if use_tqdm else range(n_epochs):
            shuffled_memory = self.shuffle_memory()
            losses = []

            self.q_network.train()
            for i in range(0, len(shuffled_memory['state']), self.q_batch_size):
                batch = self.pick_batch(shuffled_memory, (i,i+self.q_batch_size))
                loss = self.handle_q_batch(batch, pred_hands)
                losses.append(loss)

                self.q_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
                self.q_optimizer.step()
            
            self.q_network.eval()
            with torch.no_grad():
                test_loss = self.handle_q_batch(test_batch, pred_hands)
            
            if lr_schedule:
                self.q_scheduler.step(test_loss.item())
            if use_tqdm:
                t.set_description(f"Training Q-Network epoch {epoch} train loss {round(torch.mean(torch.stack(losses)).item(), 5)} test loss {round(test_loss.item(), 5)} lr {self.q_optimizer.param_groups[0]['lr']}", refresh=True)
    
    def train_hand_predictor(self, n_epochs, lr_schedule=True):
        test_episode_lengths = [episode.size for episode in list(self.episode_indices.values())[-self.hand_batch_size:]]
        end = len(self.memory['state'])
        test_batch = self.pick_batch(self.memory, (end-sum(test_episode_lengths),end))
        episode_indices = dict(itertools.islice(self.episode_indices.items(), len(self.episode_indices)-self.hand_batch_size))
        memory = self.pick_batch(self.memory, (0,end-sum(test_episode_lengths)))

        t = tqdm(range(n_epochs), desc="Training Hand Predictor")
        for epoch in t:
            episode_indices, shuffled_memory = self.shuffle_episodes(episode_indices, memory)
            episode_lengths = [episode.size for episode in episode_indices.values()]
            losses, accuracies = [], []

            j = 0
            for i in range(0, len(episode_indices), self.hand_batch_size):
                batch_lengths = episode_lengths[i:i+self.hand_batch_size]
                batch_size_sum = sum(batch_lengths)
                batch = self.pick_batch(shuffled_memory, (j,j+batch_size_sum))
                j += batch_size_sum

                self.hand_predictor.train()
                loss, acc = self.handle_hand_batch(batch, batch_lengths)
                losses.append(loss)
                accuracies.append(acc)

                self.hand_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.hand_predictor.parameters(), max_norm=1.0)
                self.hand_optimizer.step()

            self.hand_predictor.eval()
            with torch.no_grad():
                test_loss, test_acc = self.handle_hand_batch(test_batch, test_episode_lengths)
            if lr_schedule:
                self.hand_scheduler.step(test_loss.item())  # Fix: Use .item() to get a Python float
            t.set_description(f"Training Hand Predictor epoch {epoch} train loss {round(torch.mean(torch.stack(losses)).item(), 5)} test loss {round(test_loss.item(), 5)} train acc {round(np.average(accuracies), 2)} test acc {round(test_acc, 2)} lr {self.hand_optimizer.param_groups[0]['lr']}", refresh=True)
    
    def load_memory(self, memory):
        # start = time.time()
        self.memory, self.episode_indices, index = [], {}, 0
        for i, episode in enumerate(memory):
            self.episode_indices[i] = np.arange(index,index+len(episode))
            self.memory.extend(episode)
            index += len(episode)
        self.memory = self.unpack_memory(self.memory)
        self.memory['action_masks']  = self.action_masks(*self.memory['mask_dep'].values())
        self.memory['next_action_masks'] = self.action_masks(*self.memory['next_mask_dep'].values())
        # print(f"Memory loaded in {round(time.time()-start, 2)} seconds")
            
    def train_on_data(self, memory, q_epochs, hand_epochs, lr_schedule=True, use_tqdm=True):
        self.load_memory(memory)
        if hand_epochs:
            self.train_hand_predictor(hand_epochs, lr_schedule)
        if q_epochs:
            self.train_q_network(q_epochs, lr_schedule, use_tqdm)
    
    def pickle_memory(self, memory, path='stored_memories.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(memory, f)

    def train_self_play(self, n_games, q_epochs=10, hand_epochs=10, path='models/model.pth'):
        try:
            with open('stored_memories.pkl', 'rb') as f:
                memories = pickle.load(f)
            with open('call_memories.pkl', 'rb') as f:
                call_memories = pickle.load(f)
        except (FileNotFoundError, EOFError):
            memories = self.real_data
            call_memories = []
        for i in range(n_games):
            game, memory, call_memory = self.simulate_game()
            call_memories += call_memory
            memories += memory if 200 > len(game.datarows) > 50 else []
            print(f"Game {i} finished, {len(memories)} memories, {len(call_memories)} calls collected")
            self.train_on_data(memory, q_epochs, hand_epochs, lr_schedule=False)
            if i % 3 == 0 and i:
                if len(memories) > 300:
                    sample = random.sample(memories, 300)
                    self.train_on_data(sample, q_epochs*3, hand_epochs*3, lr_schedule=False)
                self.train_on_data(call_memories, q_epochs*3, 0, lr_schedule=False)
                self.pickle_memory(memories, 'stored_memories.pkl')
                self.pickle_memory(call_memories, 'call_memories.pkl')
                self.save_model(path)

    def simulate_game(self):
        def train_at_step(player):
            if len(game.hands) > 2:
                game.rotate(player)
                memories = []
                for _ in range(5):
                    game.shuffle()
                    memories.append(game.memory(player, pick_last=True))
                self.train_on_data(memories, 1, 0, lr_schedule=False, use_tqdm=False)
                game.last_indices = {}
        game = SimulatedFishGame(random.choice((6,8)))
        no_call_count = 0
        game.all_pred_hands = []
        while not game.ended():
            acted = False
            actions = {}
            saved = {}
            for player in game.players_with_cards():
                game.rotate(player)
                game_state = game.to_state()
                state = self.tensor(np.stack([game.get_state(i, game_state) for i in range(len(game.hands))]))
                mask = self.action_masks(*self.unpack_memory([game.mask_dep(len(game.hands)-1, player)]).values())
                pred_hands, action = self.act(state, mask)
                saved[player] = game.decode_pred_hands(pred_hands.cpu().detach().numpy())
                actions[player] = action
            game.all_pred_hands.append(saved)
            for player in game.players_with_cards():
                if action['call'][0] > action['call'][1] and not acted:
                    game.parse_action(action, player)
                    train_at_step(player)
                    acted = True
            if not acted and not game.ended():
                if game.turn in game.players_with_cards() and not game.asking_ended():
                    game.parse_action(actions[game.turn], game.turn)
                    train_at_step(game.turn)
                elif no_call_count < 3:
                    no_call_count += 1
                else:
                    call_confidences = {
                        player: actions[player]['call'][0]
                        for player in game.players_with_cards()
                    }
                    calling_player = max(call_confidences, key=call_confidences.get)
                    actions[calling_player]['call'] = [1,0]
                    game.parse_action(actions[calling_player], calling_player)
                    train_at_step(calling_player)
                    no_call_count = 0

        with open("sample_simulation.txt", "w") as f:
            f.writelines(game.datarows)
        memories = []
        call_memories = []
        for player in game.players:
            for _ in range(10):
                game.shuffle()
                memory, call_memory = game.memory(player, return_call_set=True)
                memories.append(memory)
                if call_memory: # check nonempty
                    call_memories.append(call_memory)
        return game, memories, call_memories

    def act(self, state, mask):  # sourcery skip: remove-redundant-if
        self.q_network.eval()
        self.hand_predictor.eval()
        with torch.no_grad():
            pred_hands = self.hand_predictor(state, mask['hands'])[-1]
            q_vals = self.q_network(pred_hands, mask)
        result = {}
        for key in q_vals.keys():
            row_data = q_vals[key][0].cpu().detach().numpy()
            random_n = random.random()
            if key == 'call_cards' and random_n < self.epsilon:
                for i in range(6):
                    row_data[i] = np.random.random(row_data[i].shape) * (row_data[i] > -9e8)
            elif (key == 'call' and random_n < self.epsilon / 10) or (key != 'call' and random_n < self.epsilon):
                row_data = np.random.random(row_data.shape) * (row_data > -9e8) # transfer masks
            result[key] = row_data
        return pred_hands, result
    
    def save_model(self, path='model.pth', q_network=True, hand_predictor=True):
        torch.save(({
                'q_network_state_dict': self.q_network.state_dict(),
                'q_optimizer_state_dict': self.q_optimizer.state_dict(),
                'q_scheduler_state_dict': (self.q_scheduler.state_dict()
                                        if hasattr(self, 'q_scheduler') else None),
            } if q_network else {}) | ({
                'hand_scheduler_state_dict': (self.hand_scheduler.state_dict()
                                              if hasattr(self, 'hand_scheduler') else None),
                'hand_predictor_state_dict': self.hand_predictor.state_dict(),
                'hand_optimizer_state_dict': self.hand_optimizer.state_dict(),
            } if hand_predictor else {}), path,)
        print(f"Model saved to {path}")
        
    def load_model(self, path='model.pth', q_network=True, hand_predictor=True):
        if not os.path.exists(path):
            return False
            
        checkpoint = torch.load(path, map_location=self.device)
        
        if hand_predictor and 'hand_predictor_state_dict' in checkpoint:
            print('loading hand predictor')
            self.hand_predictor.load_state_dict(checkpoint['hand_predictor_state_dict'])
            self.hand_optimizer.load_state_dict(checkpoint['hand_optimizer_state_dict'])
            if checkpoint['hand_scheduler_state_dict'] is not None and hasattr(self, 'hand_scheduler'):
                self.hand_scheduler.load_state_dict(checkpoint['hand_scheduler_state_dict'])
        if q_network and 'q_network_state_dict' in checkpoint:
            print('loading q vals')
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
            if checkpoint['q_scheduler_state_dict'] is not None and hasattr(self, 'q_scheduler'):
                self.q_scheduler.load_state_dict(checkpoint['q_scheduler_state_dict'])
        
        return True