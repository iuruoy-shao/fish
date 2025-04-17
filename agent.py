from verify_data import SimulatedFishGame, CALL_LEN, ASK_LEN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
import os
import pickle

class HandPrediction(nn.Module):
    def __init__(self):
        super(HandPrediction, self).__init__()
        self.rnn = nn.LSTM(54+CALL_LEN+ASK_LEN, 512, 3, batch_first=True)
        self.fc = nn.Linear(512, 8*54)
        
    def forward(self, x, mask):
        out, _ = self.rnn(x)
        return F.softmax(self.fc(out)
                         .reshape(-1,8,54)
                         .masked_fill(~mask, -1e9), dim=2)

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.rnn = nn.LSTM(54+CALL_LEN+ASK_LEN, 256, 3, batch_first=True)
        
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        
        self.to_call = nn.Linear(32, 2)
        self.pick_call_set = nn.Linear(32, 9)
        self.pick_call_cards = nn.Linear(32 + 9, 24)
        
        self.pick_person = nn.Linear(32, 4)
        self.pick_ask_set = nn.Linear(32 + 4, 9)
        self.pick_ask_card = nn.Linear(32 + 4 + 9, 6)
        
    def forward(self, x, action_masks):
        out, _ = self.rnn(x)
        h = F.relu(self.fc4(out))
        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))

        to_call = self.to_call(h)
        call_set = self.pick_call_set(h)
        call_cards = torch.reshape(self.pick_call_cards(torch.cat((h, call_set), dim=1)), (-1, 6, 4))

        ask_person = self.pick_person(h)
        ask_set = self.pick_ask_set(torch.cat((h, ask_person), dim=1))
        ask_card = self.pick_ask_card(torch.cat((h, ask_person, ask_set), dim=1))

        return {
            'call': to_call,
            'call_set': call_set.masked_fill(~action_masks['call_set'], -1e9),
            'call_cards': call_cards.masked_fill(~action_masks['call_cards'], -1e9),
            'ask_person': ask_person.masked_fill(~action_masks['ask_person'], -1e9),
            'ask_set': ask_set.masked_fill(~action_masks['ask_set'], -1e9),
            'ask_card': ask_card, 
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
            self.hand_optimizer, mode='min', factor=0.8, patience=5, 
            verbose=True, min_lr=1e-6
        )

        self.q_network = QNetwork().to(self.device)
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.q_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.q_optimizer, mode='min', factor=0.8, patience=5, 
            verbose=True, min_lr=1e-6
        )
        self.loss = nn.MSELoss()
        self.real_data = real_data

        self.gamma = 0.99    # discount factor
        self.epsilon = 0.10   # exploration rate
        self.batch_size = 4

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
    
    def action_masks(self, agent_index, hand, sets_remaining, cards_remaining):
        cards_remaining = np.array(cards_remaining)
        return {
            'hands': self.tensor(np.tile((cards_remaining > 0).reshape((-1,8,1)), (1,1,54)) 
                                 * np.tile(np.repeat(sets_remaining, 6, axis=1)[:,np.newaxis,:], (1,8,1)), as_bool=True), # cards & players still in game
            'call_set': self.tensor(sets_remaining, as_bool=True),  # the sets that remain
            'call_cards': self.tensor(np.tile(cards_remaining[:,::2] > 0, (6, 1)), as_bool=True),  # the players on the team that still have cards
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

    def shuffle_memory(self):
        indices = list(range(len(self.memory['state'])))
        random.shuffle(indices)
        
        def shuffle_item(item, idx_list):
            if isinstance(item, dict):
                return {k: shuffle_item(v, idx_list) for k, v in item.items()}
            elif isinstance(item, list):
                return [item[i] for i in idx_list]
            elif isinstance(item, torch.Tensor):
                return torch.stack([item[i] for i in idx_list])
        self.memory = {key: shuffle_item(value, indices) for key, value in self.memory.items()}
    
    def pick_batch(self, memory, indices):
        start, end = indices
        return {
            key: self.pick_batch(memory[key], (start, end))
            if isinstance(memory[key], dict)
            else memory[key][start:end]
            for key in memory.keys()
        }
    
    def handle_q_batch(self, batch):
        batch_loss = 0
        for episode in batch:
            current_q = self.q_network(self.tensor(episode['state']), episode['action_masks'])
            next_q = self.q_network(self.tensor(episode['next_state']), episode['next_action_masks'])
            batch_loss += self.q_loss(current_q, next_q, episode['action'], episode['reward'])
        return batch_loss
    
    def accuracy(self, pred_hands, episode):
        pred_hands = pred_hands.cpu().detach().numpy()
        choices = np.argmax(pred_hands, axis=1)
        one_hot = np.zeros_like(pred_hands)
        one_hot[np.arange(pred_hands.shape[0])[:,None], choices, np.arange(54)[None,:]] = 1
        cards_remaining = np.sum(np.stack(episode['mask_dep']['sets_remaining']), axis=1) * 6
        return (one_hot * episode['hands']).sum((1,2)) / cards_remaining
    
    def handle_hand_batch(self, batch):
        batch_loss = 0
        accuracies = []
        for episode in batch:
            pred_hands = self.hand_predictor(self.tensor(episode['state']), episode['action_masks']['hands'])
            accuracies += self.accuracy(pred_hands, episode).tolist()
            batch_loss += self.loss(self.tensor(episode['hands']), pred_hands)
        return batch_loss / len(batch), sum(accuracies) / len(accuracies)
    
    def train_q_network(self, n_epochs, lr_schedule=True):
        for epoch in range(n_epochs):
            random.shuffle(self.memory)
            total_loss = 0
            batch_count = 0

            self.q_network.train()
            for i in range(0, len(self.memory), self.batch_size):
                if i + self.batch_size > len(self.memory):
                    continue

                batch = self.memory[i:i+self.batch_size]
                loss = self.handle_q_batch(batch)
                total_loss += loss
                batch_count += 1

                self.q_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
                self.q_optimizer.step()
            
            train_avg_loss = total_loss / batch_count
            
            if lr_schedule:
                self.q_scheduler.step(train_avg_loss)
            
            print(f"q_network epoch {epoch}, loss {round(train_avg_loss.item(), 5)}, lr {self.q_optimizer.param_groups[0]['lr']}")
    
    def train_hand_predictor(self, n_epochs, lr_schedule=True):
        for epoch in range(n_epochs):
            random.shuffle(self.memory)
            total_loss = 0
            batch_count = 0
            accuracies = []

            self.q_network.train()
            for i in range(0, len(self.memory), self.batch_size):
                if i + self.batch_size > len(self.memory):
                    continue

                batch = self.memory[i:i+self.batch_size]
                loss, acc = self.handle_hand_batch(batch)
                total_loss += loss
                accuracies.append(acc)
                batch_count += 1

                self.hand_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.hand_predictor.parameters(), max_norm=1.0)
                self.hand_optimizer.step()
            
            train_avg_loss = total_loss / batch_count
            
            if lr_schedule:
                self.hand_scheduler.step(train_avg_loss)
            
            print(f"pred_hands epoch {epoch}, loss {round(train_avg_loss.item(), 5)}, avg acc {round(sum(accuracies) / len(accuracies), 2)}, lr {self.hand_optimizer.param_groups[0]['lr']}")
    
    def train_on_data(self, memory, q_epochs, hand_epochs, lr_schedule=True):
        self.memory = []
        for episode in memory: # [[{}]] -> [{[]}]
            unpacked = self.unpack_memory(episode)
            unpacked['action_masks']  = self.action_masks(*unpacked['mask_dep'].values())
            unpacked['next_action_masks'] = self.action_masks(*unpacked['next_mask_dep'].values())
            unpacked['hands'] = np.stack(unpacked['hands'])
            self.memory.append(unpacked)
        
        self.train_hand_predictor(hand_epochs, lr_schedule)
        self.train_q_network(q_epochs, lr_schedule)
    
    def pickle_memory(self, memory, path='memory.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(memory, f)

    def train_self_play(self, n_games, update_rate=5, epochs=10, path='models/model.pth'):
        try:
            with open('memory.pkl', 'rb') as f:
                memories = pickle.load(f)
        except (FileNotFoundError, EOFError):
            memories = self.real_data
        memories_batch = []
        for i in range(n_games):
            game, memory = self.simulate_game()
            memories_batch += memory
            memories += memory if len(game.datarows) > 50 else []
            print(f"Game {i} finished, {len(memories)} memories collected")
            if i % update_rate == 0 and i:
                self.train_on_data(memories_batch, epochs, lr_schedule=False)
                memories_batch = []
            if i % (update_rate * 3) == 0 and i:
                if len(memories):
                    self.train_on_data(memories, epochs, lr_schedule=False)
                    self.pickle_memory(memories)
                self.save_model(path)

    def simulate_game(self):
        game = SimulatedFishGame(random.choice([6,8]))
        no_call_count = 0
        while not game.ended():
            acted = False
            actions = {}
            for player in game.players_with_cards():
                game.rotate(player)
                state = self.tensor(np.stack([game.get_state(len(game.hands)-1, player, game.to_state())]))
                mask = self.action_masks(*self.unpack_memory([game.mask_dep(len(game.hands)-1, player)]).values())
                action = self.act(state, mask)
                actions[player] = action
                if action['call'][0] > action['call'][1] and not acted:
                    game.parse_action(action, player)
                    acted = True
            if not acted and not game.ended():
                if game.turn in game.players_with_cards() and not game.asking_ended():
                    game.parse_action(actions[game.turn], game.turn)
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
                    no_call_count = 0

        with open("sample_simulation.txt", "w") as f:
            f.writelines(game.datarows)
        memories = []
        for player in game.players: # TODO: multithread this for cuda
            for _ in range(10):
                game.shuffle()
                memories.append(game.memory(player))
        return game, memories

    def act(self, state, mask):
        self.q_network.eval()
        with torch.no_grad():
            q_vals = self.q_network(state, mask)
        result = {}
        for key in q_vals.keys():
            row_data = q_vals[key][0].cpu().detach().numpy()
            if random.random() < self.epsilon:
                if key != 'call_cards':
                    row_data = np.random.random(row_data.shape) * (row_data > -9e8) # transfer masks
                else:
                    for i in range(6):
                        row_data[i] = np.random.random(row_data[i].shape) * (row_data[i] > -9e8)
            result[key] = row_data
        return result
    
    def save_model(self, path='model.pth'):
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'q_scheduler_state_dict': self.q_scheduler.state_dict() if hasattr(self, 'q_scheduler') else None,
            'epsilon': self.epsilon,
            'gamma': self.gamma,
        }, path)
        print(f"Model saved to {path}")
        
    def load_model(self, path='model.pth'):
        if not os.path.exists(path):
            return False
            
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        
        if checkpoint['q_scheduler_state_dict'] is not None and hasattr(self, 'q_scheduler'):
            self.q_scheduler.load_state_dict(checkpoint['q_scheduler_state_dict'])
            
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.gamma = checkpoint.get('gamma', self.gamma)
        
        return True