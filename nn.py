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

        self.pick_person = nn.Linear(64, 1)
        self.pick_ask_set = nn.Linear(64 + 1, 9)
        self.pick_ask_card = nn.Linear(64 + 1 + 9, 6)
        
    def forward(self, x, action_masks):
        x = torch.flatten(x, 1)
        x = self.fc3(self.fc2(self.fc1(x)))
        to_call = self.to_call(x)
        call_set = self.pick_call_set(x)
        call_cards = torch.reshape(self.pick_call_cards(torch.cat((x, call_set), 1)), (-1,6,4))
        ask_person = self.pick_person(x)
        ask_set = self.pick_ask_set(torch.cat((x, ask_person), 1))
        ask_card = self.pick_ask_card(torch.cat((x, ask_person, ask_set), 1))

        return { # masking & normalizing
            'call': F.sigmoid(to_call),
            'call_set': F.softmax(call_set.masked_fill(action_masks['call_set'], -float('inf'))),
            'call_cards': F.softmax(call_cards.masked_fill(action_masks['call_cards'], -float('inf'))), 
            'ask_person': F.softmax(ask_person.masked_fill(action_masks['ask_person'], -float('inf'))),
            'ask_set': F.softmax(ask_set.masked_fill(action_masks['ask_set'], -float('inf'))),
            'ask_card': F.softmax(ask_card), 
        }

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, memory):
        # self.agent_index = mask_dependencies['agent_index'],
        # self.hand = mask_dependencies['hand'],
        # self.sets_remaining = mask_dependencies['sets_remaining'],

        self.memory = memory
        self.device = torch.device("mps" if torch.backends.mps.is_available() 
                                   else "cuda" if torch.cuda.is_available() 
                                   else "cpu")
        
        self.q_network = QNetwork().to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters())
        
        self.gamma = 0.99    # discount factor
        self.epsilon = 0.1   # exploration rate
        self.batch_size = 32

    def to_tensor(self, x, as_bool=False):
        if as_bool:
            return torch.BoolTensor(x).to(self.device)
        return torch.FloatTensor(x).to(self.device)

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
    
    def action_masks(self, agent_index, hand, sets_remaining, cards_remaining):
        cards_remaining = np.array(cards_remaining)
        return {
            'call_set': self.to_tensor(sets_remaining, as_bool=True), # the sets that remain
            'call_cards': self.to_tensor([np.tile(np.array(cards_remaining)[:,index%2::2] > 0, (1, 6, 1)) 
                                          for index in agent_index], as_bool=True), # the players on the team that still have cards
            'ask_person': self.to_tensor([np.array(cards_remaining)[:,(index+1)%2::2] > 0 
                                          for index in agent_index], as_bool=True), # the players on the opposing team that still have cards
            'ask_set': self.to_tensor(np.sum(hand, axis=1) > 0, as_bool=True) # the sets that the player holds
        }
    
    def unpack_batch(self, batch):
        result = {}
        if not batch:
            return result
        keys = batch[0].keys()
        
        for key in keys:
            if isinstance(batch[0][key], dict):
                result[key] = self.unpack_batch([item[key] for item in batch])
            else:
                result[key] = [item[key] for item in batch]
        return result
    
    def train(self):    
        batch = self.unpack_batch(random.sample(self.memory, self.batch_size))
        current_q = self.q_network(self.to_tensor(batch['state']), self.action_masks(*batch['mask_dep'].values()))
        next_q = self.q_network(self.to_tensor(batch['next_state']), self.action_masks(*batch['mask_dep'].values()))
        loss = self.q_loss(current_q, next_q, batch['action'], batch['reward'])

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()