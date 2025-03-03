import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(200 * 54, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)

        self.to_call = nn.Linear(64, 1) 
        self.pick_call_set = nn.Linear(64, 9)
        self.pick_call_cards = nn.Linear(64 + 9, 24) # will pick top value for a single section of len 4

        self.pick_person = nn.Linear(64, 1)
        self.pick_ask_set = nn.Linear(64 + 1, 9)
        self.pick_ask_card = nn.Linear(64 + 1 + 9, 6)

        self.pick_pass = nn.Linear(64, 3)
        
    def forward(self, x, action_masks):
        x = torch.flatten(x)
        x = self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(x)))))
        to_call = self.to_call(x)
        call_set = self.pick_call_set(x)
        call_cards = torch.reshape(self.pick_call_card(torch.cat((x, call_set), 1)),(6,4))
        ask_person = self.pick_person(x)
        ask_set = self.pick_ask_set(torch.cat((x, ask_person), 1))
        ask_card = self.pick_ask_card(torch.cat((x, ask_person, ask_set), 1))
        pick_pass = self.pick_pass(x)

        return { # masking & normalizing
            'call': F.sigmoid(to_call),
            'call_set': F.softmax(call_set.masked_fill(action_masks['call_set'], -float('inf'))),
            'call_cards': F.softmax(call_cards.masked_fill(action_masks['call_cards'], -float('inf')), dim=0), 
            'ask_person': F.softmax(ask_person.masked_fill(action_masks['ask_person'], -float('inf'))),
            'ask_set': F.softmax(ask_set.masked_fill(action_masks['ask_set'], -float('inf'))),
            'ask_card': F.softmax(ask_card), 
            'pick_pass': F.softmax(pick_pass.masked_fill(action_masks['pick_pass'], -float('inf')))
        }

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, action_size):
        self.agent_index = None,
        self.hand = np.array() # 9x6
        self.sets_remaining = np.array() # 6
        self.card_count = np.array() # 8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = QNetwork(action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters())
        self.memory = deque(maxlen=10000)
        
        self.gamma = 0.99    # discount factor
        self.epsilon = 0.1   # exploration rate
        self.batch_size = 64
    
    def remember(self, state, action, reward, next_state, done): # TODO: understand this
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if random.random() < self.epsilon: # explore
            
        else: # exploit
        
        # Convert state to tensor and add batch dimension
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.q_network(state) # return action valeus
        return torch.argmax(action_values).item()
    
    def update_values(self): # update hands, sets_remaining, players_with_cards, etc.
        pass

    def q_loss(self, ):
        pass
    
    def action_masks(self):
        return {
            'call_set': self.sets_remaining, # the sets that remain
            'call_cards': np.tile(self.cards_remaining[self.agent_index%2::2] > 0, (6, 1)), # the players on the team that still have cards
            'ask_person': self.cards_remaining[(self.agent_index+1)%2::2] > 0, # the players on the opposing team that still have cards
            'ask_set': (np.sum(self.hand, axis=1) > 0).astype(int), # the sets that the player holds
            'pick_pass': self.cards_remaining[(self.agent_index+1)%2::2] > 0 # the players on the team that still have cards
        }
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([i[0] for i in batch]).to(self.device)
        actions = torch.LongTensor([i[1] for i in batch]).to(self.device)
        rewards = torch.FloatTensor([i[2] for i in batch]).to(self.device)
        next_states = torch.FloatTensor([i[3] for i in batch]).to(self.device)
        dones = torch.FloatTensor([i[4] for i in batch]).to(self.device)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.q_network(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Example usage
action_size = 2   # number of possible actions
agent = QLearningAgent(action_size)