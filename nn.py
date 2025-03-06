import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(200 * 54, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)

        self.to_call = nn.Linear(64, 1) 
        self.pick_call_set = nn.Linear(64, 9)
        self.pick_call_cards = nn.Linear(64 + 9, 24) # will pick top value for a single section of len 4

        self.pick_person = nn.Linear(64, 4)
        self.pick_ask_set = nn.Linear(64 + 4, 9)
        self.pick_ask_card = nn.Linear(64 + 4 + 9, 6)
        
    def forward(self, x, action_masks):
        x = torch.flatten(x, 1)
        x = self.fc3(self.fc2(self.fc1(x)))
        to_call = self.to_call(x)
        call_set = self.pick_call_set(x)
        call_cards = torch.reshape(self.pick_call_cards(torch.cat((x, call_set), 1)), (-1, 6, 4))
        ask_person = self.pick_person(x)
        ask_set = self.pick_ask_set(torch.cat((x, ask_person), 1))
        ask_card = self.pick_ask_card(torch.cat((x, ask_person, ask_set), 1))

        return {  # masking & normalizing
            'call': torch.sigmoid(to_call),
            'call_set': F.softmax(call_set.masked_fill(~action_masks['call_set'], -1e9), dim=1),
            'call_cards': F.softmax(call_cards.masked_fill(~action_masks['call_cards'], -1e9), dim=2),
            'ask_person': F.softmax(ask_person.masked_fill(~action_masks['ask_person'], -1e9), dim=1),
            'ask_set': F.softmax(ask_set.masked_fill(~action_masks['ask_set'], -1e9), dim=1),
            'ask_card': F.softmax(ask_card), 
        }

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, memory):
        self.memory = memory
        self.device = torch.device("mps" if torch.backends.mps.is_available() 
                                   else "cuda" if torch.cuda.is_available() 
                                   else "cpu")
        
        self.q_network = QNetwork().to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, 
            verbose=True, min_lr=1e-6
        )
        
        self.gamma = 0.99    # discount factor
        self.epsilon = 0.1   # exploration rate
        self.batch_size = 32

    def tensor(self, x, as_bool=False):
        if as_bool:
            return torch.BoolTensor(x).to(self.device)
        return torch.FloatTensor(x).to(self.device)

    def input_actions():
        pass
        
    def act(self, state):  # TODO: complete
        # Convert state to tensor and add batch dimension
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.q_network(state)  # return action values
        if random.random() < self.epsilon:  # explore
            pass
        else:  # exploit
            pass

    def max_q(self, action, i):
        if action['call'][i] > .5:
            return torch.max(action['call_cards'][i]).item()
        return torch.max(action['ask_card'][i]).item()

    def q_loss(self, action, next_action, player_action, reward):
        loss = lambda prev, max_q_next, player, reward: F.mse_loss(sum(prev * player), 
                                                                   reward + self.gamma * max_q_next)
        return sum(
            sum(
                loss(
                    action[act][i],
                    self.max_q(next_action, i),
                    self.tensor(player_action[act][i]),
                    self.tensor(reward[i]),
                )
                for act in player_action.keys()
                if player_action[act][i] is not None
            )
            for i in range(len(reward))
        )
    
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
    
    def train(self, n_epochs):
        for epoch in range(n_epochs):
            random.shuffle(self.memory)
            total_loss = 0
            batch_count = 0

            for i in range(0, len(self.memory), self.batch_size):
                if i + self.batch_size > len(self.memory):
                    continue

                batch = self.unpack_batch(self.memory[i:i+self.batch_size])
                current_q = self.q_network(self.tensor(batch['state']), self.action_masks(*batch['mask_dep'].values()))
                next_q = self.q_network(self.tensor(batch['next_state']), self.action_masks(*batch['mask_dep'].values()))
                
                loss = self.q_loss(current_q, next_q, batch['action'], batch['reward'])
                total_loss += loss.item()
                batch_count += 1

                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Calculate average loss for this epoch
            avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
            
            # Update learning rate based on loss
            self.scheduler.step(avg_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"epoch {epoch}, loss {avg_loss:.5f}, lr {current_lr}")