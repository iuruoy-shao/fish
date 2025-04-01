from verify_data import SimulatedFishGame
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(200 * 54, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.dropout = nn.Dropout(0.1)
        
        self.call_head = nn.Linear(256, 64)
        self.ask_head = nn.Linear(256, 64)

        self.to_call = nn.Linear(64, 2) 
        self.pick_call_set = nn.Linear(64, 9)
        self.pick_call_cards = nn.Linear(64 + 9, 24) # will pick top value for a single section of len 4

        self.pick_person = nn.Linear(64, 4)
        self.pick_ask_set = nn.Linear(64 + 4, 9)
        self.pick_ask_card = nn.Linear(64 + 4 + 9, 6)
        
    def forward(self, x, action_masks):
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        
        call_head = self.dropout(F.relu(self.call_head(x)))
        ask_head = self.dropout(F.relu(self.ask_head(x)))

        to_call = self.to_call(call_head)
        call_set = self.pick_call_set(call_head)
        call_cards = torch.reshape(self.pick_call_cards(torch.cat((call_head, call_set), 1)), (-1, 6, 4))
        
        ask_person = self.pick_person(ask_head)
        ask_set = self.pick_ask_set(torch.cat((ask_head, ask_person), 1))
        ask_card = self.pick_ask_card(torch.cat((ask_head, ask_person, ask_set), 1))

        return {  # masking & normalizing
            'call': F.softmax(to_call, dim=1),
            'call_set': F.softmax(call_set.masked_fill(~action_masks['call_set'], -1e9), dim=1),
            'call_cards': F.softmax(call_cards.masked_fill(~action_masks['call_cards'], -1e9), dim=2),
            'ask_person': F.softmax(ask_person.masked_fill(~action_masks['ask_person'], -1e9), dim=1),
            'ask_set': F.softmax(ask_set.masked_fill(~action_masks['ask_set'], -1e9), dim=1),
            'ask_card': F.softmax(ask_card), 
        }

# Q-Learning Agent
class QLearningAgent:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() 
                                   else "cuda" if torch.cuda.is_available() 
                                   else "cpu")
        
        self.q_network = QNetwork().to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=5, 
            verbose=True, min_lr=1e-6
        )
        self.loss = nn.MSELoss()
        
        self.gamma = 0.99    # discount factor
        self.epsilon = 0.1   # exploration rate
        self.batch_size = 32

    def tensor(self, x, as_bool=False):
        if as_bool:
            return torch.BoolTensor(x).to(self.device)
        return torch.FloatTensor(x).to(self.device)

    def max_q(self, action, i):
        if torch.argmax(action['call'][i]) == 0:
            return torch.max(action['call_cards'][i])
        return torch.max(action['ask_card'][i])
    
    def q_vals(self, prev, max_q_next, player, reward): 
        return torch.sum(prev * player), reward + self.gamma * max_q_next

    def q_loss(self, action, next_action, player_action, reward):
        current_q = []
        target_q = []
        for i in range(len(reward)):
            for act in player_action.keys():
                if player_action[act][i] is None:
                    continue # skip Nones
                this_reward = self.max_q(next_action, i)
                this_max_q = self.max_q(action, i)
                this_action = action[act][i]
                this_player_action = self.tensor(player_action[act][i])
                if act == 'call_cards':
                    for row in range(6):
                        current, target = self.q_vals(this_action[row], this_max_q, this_player_action[row], this_reward)
                        current_q.append(current)
                        target_q.append(target)
                else:
                    current, target = self.q_vals(this_action, this_max_q, this_player_action, this_reward)
                    current_q.append(current)
                    target_q.append(target)
        return self.loss(torch.stack(current_q), torch.stack(target_q))
    
    def action_masks(self, agent_index, hand, sets_remaining, cards_remaining):
        cards_remaining = np.array(cards_remaining)
        return {
            'call_set': self.tensor(sets_remaining, as_bool=True),  # the sets that remain
            'call_cards': self.tensor([np.tile(np.array(cards_remaining)[i, index % 2::2] > 0, (6, 1)) 
                                       for i, index in enumerate(agent_index)], as_bool=True),  # the players on the team that still have cards
            'ask_person': self.tensor([np.array(cards_remaining)[i, (index + 1) % 2::2] > 0 
                                       for i, index in enumerate(agent_index)], as_bool=True),  # the players on the opposing team that still have cards
            'ask_set': self.tensor(np.sum(hand, axis=2) > 0, as_bool=True)  # the sets that the player holds
        }
    
    def unpack_batch(self, batch):
        return {
            key: (
                self.unpack_batch([item[key] for item in batch])
                if isinstance(batch[0][key], dict)
                else [item[key] for item in batch]
            )
            for key in batch[0].keys()
        }
    
    def handle_batch(self, batch):
        current_q = self.q_network(self.tensor(batch['state']), self.action_masks(*batch['mask_dep'].values()))
        next_q = self.q_network(self.tensor(batch['next_state']), self.action_masks(*batch['mask_dep'].values()))
                
        return self.q_loss(current_q, next_q, batch['action'], batch['reward'])
    
    def train_on_data(self, train_memory, test_memory, n_epochs):
        self.memory = train_memory
        
        for epoch in range(n_epochs):
            random.shuffle(self.memory)
            total_loss = 0
            batch_count = 0

            for i in range(0, len(self.memory), self.batch_size):
                if i + self.batch_size > len(self.memory):
                    continue

                batch = self.unpack_batch(self.memory[i:i+self.batch_size])
                loss = self.handle_batch(batch)
                total_loss += loss
                batch_count += 1

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            train_avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
            
            with torch.no_grad():
                batch = self.unpack_batch(test_memory)
                test_loss = self.handle_batch(batch)
            
            self.scheduler.step(train_avg_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"epoch {epoch}, train loss {train_avg_loss:.5f}, test loss {test_loss:.5f}, lr {current_lr}")
    
    def train_self_play(self):
        game = SimulatedFishGame(random.choice([6,8]))
        states = []
        action_masks = []
        for player in game.players_with_cards():
            game.rotate(player)
            states.append(game.to_state())
            action_masks.append(game.mask_dep(-1, player))
        print(self.unpack_batch(states), self.unpack_batch(action_masks))

    def act(self, state, mask):
        q_vals = self.q_network(state, mask)
        for key in q_vals.keys():
            q_vals[key] = q_vals[key].cpu().detach().numpy()
            if random.random() < self.epsilon:
                q_vals[key] = np.random.dirichlet(np.ones(q_vals[key]))

    def save_model(self, path='model.pth'):
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'epsilon': self.epsilon,
            'gamma': self.gamma,
        }, path)
        print(f"Model saved to {path}")
        
    def load_model(self, path='model.pth'):
        if not os.path.exists(path):
            return False
            
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] is not None and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.gamma = checkpoint.get('gamma', self.gamma)
        
        return True