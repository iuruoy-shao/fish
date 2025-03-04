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
        self.fc2 = nn.Linear(1024, 256)
        self.fc5 = nn.Linear(256, 64)

        self.to_call = nn.Linear(64, 1) 
        self.pick_call_set = nn.Linear(64, 9)
        self.pick_call_cards = nn.Linear(64 + 9, 24) # will pick top value for a single section of len 4

        self.pick_person = nn.Linear(64, 1)
        self.pick_ask_set = nn.Linear(64 + 1, 9)
        self.pick_ask_card = nn.Linear(64 + 1 + 9, 6)
        
    def forward(self, x, action_masks):
        x = torch.flatten(x)
        x = self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(x)))))
        to_call = self.to_call(x)
        call_set = self.pick_call_set(x)
        call_cards = torch.reshape(self.pick_call_card(torch.cat((x, call_set), 1)),(6,4))
        ask_person = self.pick_person(x)
        ask_set = self.pick_ask_set(torch.cat((x, ask_person), 1))
        ask_card = self.pick_ask_card(torch.cat((x, ask_person, ask_set), 1))

        return { # masking & normalizing
            'call': F.sigmoid(to_call),
            'call_set': F.softmax(call_set.masked_fill(action_masks['call_set'], -float('inf'))),
            'call_cards': F.softmax(call_cards.masked_fill(action_masks['call_cards'], -float('inf')), dim=0), 
            'ask_person': F.softmax(ask_person.masked_fill(action_masks['ask_person'], -float('inf'))),
            'ask_set': F.softmax(ask_set.masked_fill(action_masks['ask_set'], -float('inf'))),
            'ask_card': F.softmax(ask_card), 
        }

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, mask_dependencies, memory):
        self.agent_index = mask_dependencies['agent_index'],
        self.hand = mask_dependencies['hand'],
        self.sets_remaining = mask_dependencies['sets_remaining'],
        
        self.memory = memory
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = QNetwork().to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters())
        
        self.gamma = 0.99    # discount factor
        self.epsilon = 0.1   # exploration rate
        self.batch_size = 32

    def input_actions():
        pass
        
    def act(self, state): #TODO: complete
        # Convert state to tensor and add batch dimension
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.q_network(state) # return action valeus
        if random.random() < self.epsilon: # explore
            pass
        else: # exploit
            pass

    def q_loss(self, action, next_action, player_action, reward):
        loss = lambda prev, next, player, reward: F.mse_loss(sum(prev * player), torch.tensor([reward + self.gamma * torch.max(next).item()]))
        return sum(loss(action[act], next_action[act], player_action[act], reward) 
                   for act in player_action.keys() 
                   if player_action[act] is not None)
    
    def action_masks(self):
        return {
            'call_set': self.sets_remaining, # the sets that remain
            'call_cards': np.tile(self.cards_remaining[self.agent_index%2::2] > 0, (6, 1)), # the players on the team that still have cards
            'ask_person': self.cards_remaining[(self.agent_index+1)%2::2] > 0, # the players on the opposing team that still have cards
            'ask_set': (np.sum(self.hand, axis=1) > 0).astype(int), # the sets that the player holds
        }
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        state = [], action = [], reward = [], next_state = [], dones = []
        for row in batch:
            state.append(row['state'])
            action.append(row['action'])
            reward.append(row['reward'])
            next_state.append(row['next_state'])
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        
        current_q = self.q_network(state)
        next_q = self.q_network(next_state)
        loss = self.q_loss(current_q, next_q, action, reward)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

agent = QLearningAgent()