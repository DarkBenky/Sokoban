import random
import os
import numpy as np
import json
import sys
import pickle
import hashlib
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
# import pprint
from stable_baselines3 import DQN
import pygame
# from numba import jit

INVALID_MOVE_PENALTY = -2
WIN_REWARD = 10
BOX_PUSH_REWARD = 1.5
BOX_CLOSE_TO_GOAL_REWARD = 2.5
LEARNING_RATE = 0.001
BOX_NEAR_BARRIER_PENALTY = 0.125
PROBABILITY_OF_USING_HISTORY = 0.5
EPOCH = 100
TRAINING_STEPS = 1_000

from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))



def load_function_from_json(folder_path = 'maps' , map_name = None):
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    if not files:
        raise FileNotFoundError("No JSON files found in the specified folder.")

    # Select a random JSON file
    if map_name != None:
        file_path = os.path.join(folder_path, map_name)
    else:
        random_file = random.choice(files)
        file_path = os.path.join(folder_path, random_file)

    # Load data from the selected file
    data = json.load(open(file_path))
    
    # Extract information from the data
    barriers = data.get('barriers', [])
    player_x, player_y = data.get('player', [0, 0])
    box = data.get('box', [])
    goals = data.get('goals', [])
    
    if len(box) == 2:
        box = [box]
    if len(goals) == 2:
        goals = [goals]
    if len(barriers) == 2:
        barriers = [barriers]
    
    return barriers, player_x, player_y, box, goals


class SokobanEnv:
    def __init__(self, playerXY=(0, 0), barriers=[], boxes=[], goals=[]):
        self.playerXY = np.array(playerXY)
        self.boxes = np.array(boxes)
        self.goals = np.array(goals)
        self.barriers = np.array(barriers)
        self.arena = self.construct_arena()
        self.obs = self.construct_obs()      
        self.action_space = np.array([0, 1, 2, 3])
        self.last_box_move = 0
        self.step_count = 0
              
    def generate_heatmap(self, size=(10, 10), coordinates=(0, 0), player=True):
        heatmap = np.zeros(size)
        for i in range(size[0]):
            for j in range(size[1]):
                if i == coordinates[0] and j == coordinates[1] and player:
                    heatmap[i][j] = 0
                    continue
                distance = np.linalg.norm(np.array([i, j]) - np.array(coordinates))
                heatmap[i][j] = 1 / ((distance * distance) + 1)
        
        for barrier in self.barriers:
            heatmap[barrier[0]][barrier[1]] = INVALID_MOVE_PENALTY
            
        return heatmap

    def construct_obs(self):
        # Initialize empty heatmaps for boxes and goals
        box_heatmap = np.zeros((10, 10))
        goal_heatmap = np.zeros((10, 10))
        
        # Calculate distance from each position to boxes and sum them up
        for box in self.boxes:
            box_heatmap += self.generate_heatmap(coordinates=box, player=False)
        
        # Calculate distance from each position to goals and sum them up
        for goal in self.goals:
            goal_heatmap += self.generate_heatmap(coordinates=goal, player=False)
        
        # Calculate player heatmap
        player_heatmap = self.generate_heatmap(coordinates=self.playerXY, player=True)
        
        # Penalize positions with barriers
        for barrier in self.barriers:
            box_heatmap[barrier[0]][barrier[1]] = INVALID_MOVE_PENALTY
            goal_heatmap[barrier[0]][barrier[1]] = INVALID_MOVE_PENALTY
            player_heatmap[barrier[0]][barrier[1]] = INVALID_MOVE_PENALTY
        
        # Normalize heatmaps
        box_heatmap /= np.max(box_heatmap)
        goal_heatmap /= np.max(goal_heatmap)
        player_heatmap /= np.max(player_heatmap)
        
        # Calculate the distance of each position to the nearest box
        min_box_distance = np.min(box_heatmap, axis=(0, 1))
        
        # Penalize positions closer to the finish without nearby boxes
        penalty_factor = 0.5  # Adjust this factor as needed
        goal_heatmap -= penalty_factor * (1 - min_box_distance)
        
        # Combine heatmaps to create final observation
        final_obs = (box_heatmap + goal_heatmap + player_heatmap) / 3
        
        dxdy = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        for box in self.boxes:
            final_obs[box[0]][box[1]] += BOX_PUSH_REWARD
            if np.any(np.all(self.goals == box, axis=1)):
                final_obs[box[0]][box[1]] += BOX_CLOSE_TO_GOAL_REWARD
            for dx, dy in dxdy:
                new_position = box + np.array([dx, dy])
                if self.is_valid_move(new_position) and not self.is_box_collision(new_position):
                    final_obs[new_position[0]][new_position[1]] += BOX_CLOSE_TO_GOAL_REWARD / 4
            
            # Check if the box move is close to barriers and penalize
            for barrier in self.barriers:
                for dx, dy in dxdy:
                    near_barrier = box + np.array([dx, dy])
                    if np.all(near_barrier == barrier):
                        final_obs[box[0]][box[1]] -= BOX_NEAR_BARRIER_PENALTY
        
        # Check if box to final is in straight line and add reward
        # for box in self.boxes:
        #     for goal in self.goals:
        #         if box[0] == goal[0] or box[1] == goal[1]:
        #             if box[0] == goal[0]:
        #                 if box[1] < goal[1]:
        #                     for i in range(box[1] + 1, goal[1]):
        #                         if final_obs[box[0]][i] < 0:
        #                             final_obs[box[0]][i] += 0.25
        #                 else:
        #                     for i in range(goal[1] + 1, box[1]):
        #                         if final_obs[box[0]][i] < 0:
        #                             final_obs[box[0]][i] += 0.25
        #             else:
        #                 if box[0] < goal[0]:
        #                     for i in range(box[0] + 1, goal[0]):
        #                         if final_obs[i][box[1]] < 0:
        #                             final_obs[i][box[1]] += 0.25
        #                 else:
        #                     for i in range(goal[0] + 1, box[0]):
        #                         if final_obs[i][box[1]] < 0:
        #                             final_obs[i][box[1]] += 0.25
        
        # Encode player position 
        final_obs[self.playerXY[0]][self.playerXY[1]] = -1
        
        return final_obs

    def construct_arena(self):
        arena = np.zeros((10, 10))

        for barrier in self.barriers:
            arena[barrier[0]][barrier[1]] = 1

        for goal in self.goals:
            arena[goal[0]][goal[1]] = 2

        for box in self.boxes:
            arena[box[0]][box[1]] = 3

        arena[self.playerXY[0]][self.playerXY[1]] = 4

        return arena
    


    def move(self, direction , execute = True):
        self.step_count += 1
        new_player_position = self.playerXY.copy()

        if direction == 'W':
            new_player_position[0] -= 1
        elif direction == 'S':
            new_player_position[0] += 1
        elif direction == 'A':
            new_player_position[1] -= 1
        elif direction == 'D':
            new_player_position[1] += 1

        # Check if a box is being pushed
        box_index = np.where(np.all(self.boxes == new_player_position, axis=1))
        if box_index[0].size > 0:
            new_box_position = self.boxes[box_index][0] + (new_player_position - self.playerXY)
            if self.is_valid_move(new_box_position) and not self.is_box_collision(new_box_position):
                # Update box position
                if execute:
                    self.boxes[box_index] = new_box_position
                    self.playerXY = new_player_position
                    self.last_box_move = self.step_count
                return True
            else:
                return False
        else:
            # Update player position after checking for box movement
            if self.is_valid_move(new_player_position):
                if execute:
                    self.playerXY = new_player_position
                return True
        return False

    def is_valid_move(self, position):
        return (0 <= position[0] < self.arena.shape[0] and
                0 <= position[1] < self.arena.shape[1] and
                self.arena[position[0]][position[1]] != 1)

    def is_win(self):
        return all(np.any(np.all(self.boxes == goal, axis=1) for goal in self.goals))

    def is_box_collision(self, new_box_position):
        # Check if the new box position collides with any other boxes or barriers
        if len(self.barriers) > 0:
            barrier_collision = np.any(np.all(self.barriers == new_box_position, axis=1))
        else:
            barrier_collision = False

        return (
            np.any(np.all(self.boxes == new_box_position, axis=1)) or
            barrier_collision
        )
    
    def valid_moves(self):
        possible_moves = ['W', 'S', 'A', 'D']
        valid_moves = []

        for move in possible_moves:
            if self.move(move , execute = False):
                valid_moves.append(1)
            else:
                valid_moves.append(0)
        return valid_moves
    
    def get_direction_from_action(self, action):
        # Convert action index to corresponding direction
        if action == 0:
            return 'W'
        elif action == 1:
            return 'S'
        elif action == 2:
            return 'A'
        elif action == 3:
            return 'D'

    
    def step(self, action):
        # Take a step in the environment based on the selected action
        direction = self.get_direction_from_action(action)

        previous_obs = self.obs.copy()
        previous_box_positions = self.boxes.copy()
        success = self.move(direction)

        done = False
        reward = 0

        if success:
            self.arena = self.construct_arena()
            self.obs = self.construct_obs()
            if self.is_win():
                reward = WIN_REWARD
                print('WIN')
                done = True
                info = {'valid':True}
            else:
                if np.any(previous_box_positions != self.boxes):
                    reward += BOX_PUSH_REWARD
                self.obs = self.construct_obs()
                reward += previous_obs[self.playerXY[0]][self.playerXY[1]]
                done = False  # Set done to False since the game is not yet finished
                info = {'valid':True}  # Set info to an empty dictionary
                return self.obs, reward, done, info  # Return the updated observation, reward, done, and info values
        else:
            reward = INVALID_MOVE_PENALTY
            info = {'valid':False}
        
        return self.obs, reward, done, info
    
    def reset(self):
        barriers, player_x, player_y, box, goals = load_function_from_json('maps')
        
        self.playerXY = np.array((player_x, player_y))
        self.barriers = np.array(barriers)
        self.boxes = np.array(box)
        self.goals = np.array(goals)
        self.arena = self.construct_arena()
        self.obs = self.construct_obs()

        return self.arena

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
   
class PPOModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPOModel, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        
        # Policy network
        self.conv1 = nn.Conv2d(input_dim[0], 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1)
        self.conv_out_size = self._get_conv_output_dim(self.input_dim)
        
        self.fc_policy = nn.Linear(self.conv_out_size, 512)
        self.fc_action = nn.Linear(512, output_dim)
        
        # Value network
        self.fc_value = nn.Linear(self.conv_out_size, 512)
        self.fc_value_out = nn.Linear(512, output_dim//4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        
        x_policy = x.view(-1, self.conv_out_size)
        
        x_policy = F.relu(self.fc_policy(x_policy))
        action_probs = F.softmax(self.fc_action(x_policy), dim=-1)
        
        x_value = x.view(-1, self.conv_out_size)  # Reshape for value network
        x_value = F.relu(self.fc_value(x_value))
        value = self.fc_value_out(x_value)
        
        return action_probs, value

    def _get_conv_output_dim(self, shape):
        dummy_input = torch.zeros(1, *shape)
        dummy_output = self._forward_conv(dummy_input)
        return int(torch.prod(torch.tensor(dummy_output.shape)))

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        return x


class PPOAgent:
    def __init__(self, env: SokobanEnv, gamma=0.99, epsilon=0.2, clip_value=0.2 , batch = 1024):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PPOModel((batch, 10, 10), len(self.env.action_space)*batch).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss_fn_policy = nn.CrossEntropyLoss()
        self.loss_fn_value = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip_value = clip_value
        self.batch = batch

    def choose_action(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
            state_tensor = state_tensor.repeat(self.batch, 1, 1)
            action_prob, _ = self.model(state_tensor)
            action_prob = action_prob.view(self.batch, 4)
            action_prob = torch.mean(action_prob, 0)
            # print(f'{action_prob.shape=}')
            action = torch.multinomial(action_prob, 1).item()
            return action

    def learn(self, transitions):
        state_batch = torch.tensor(np.array([t.state for t in transitions]), dtype=torch.float32).to(self.device)
        action_batch = torch.tensor([t.action for t in transitions], dtype=torch.long).to(self.device)
        reward_batch = torch.tensor([t.reward for t in transitions], dtype=torch.float32).to(self.device)
        next_state_batch = torch.tensor(np.array([t.next_state for t in transitions]), dtype=torch.float32).to(self.device)
        done_batch = torch.tensor([t.done for t in transitions], dtype=torch.float32).to(self.device)
        
        action_probs_old, value_old = self.model(state_batch.unsqueeze(0))
        
        # convert probabilities
        action_probs_old = F.softmax(action_probs_old, dim=-1)
        
        action_probs_old = action_probs_old.view(self.batch, 4)
        action_probs_old = torch.multinomial(action_probs_old, 1, replacement=True)
        
        action_probs_old = action_probs_old.gather(0, action_batch.unsqueeze(1))
        
        _, value_next = self.model(next_state_batch)
        
        returns = self._compute_returns(reward_batch, value_next, done_batch)
        advantages = self._compute_advantages(reward_batch, value_next, value_old, done_batch)
        
        for _ in range(EPOCH):
            action_probs, value = self.model(state_batch)
            # if (action_probs <= 0).any():
            #     print("Warning: Zero or negative values found in action probabilities.")
            action_probs = F.softmax(action_probs, dim=-1)
            action_probs = action_probs.view(self.batch, 4)
            action_probs = torch.multinomial(action_probs, 1, replacement=True)
            action_probs = action_probs.gather(0, action_batch.unsqueeze(1))
            
             # Policy loss
            EPSILON = 1e-8  # Small epsilon value to prevent division by zero

            ratio = torch.exp(torch.log(action_probs + EPSILON) - torch.log(action_probs_old + EPSILON))
            # print(f'Ratio shape: {ratio.shape}, Advantages shape: {advantages.shape}')
            # print(f'Ratio values: {ratio}, Advantages values: {advantages}')
            policy_loss = -torch.mean(torch.min(ratio * advantages.unsqueeze(1), torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages.unsqueeze(1)))
            print(f'Policy loss value: {policy_loss.item()}')
                
            # Value loss
            value_loss = self.loss_fn_value(value, returns)
            print(f'Value loss value: {value_loss.item()}')
            
            # Total loss
            loss = policy_loss + value_loss
            
            print(f'Total loss value: {loss.item()}', end='\n\n')
            
            # Gradient descent
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
    
    def _compute_returns(self, rewards, value_next, dones):
        returns = torch.zeros_like(rewards)
        running_return = value_next.detach()
        for t in reversed(range(len(rewards))):
            # Make sure running_return is a scalar if it's a single value
            returns[t] = running_return[:,t]
        return returns.detach()
    
    def _compute_advantages(self, rewards, value_next, value_old, dones):
        advantages = torch.zeros_like(rewards)
        running_advantage = torch.tensor(0.0).to(self.device)
        for t in reversed(range(len(rewards))):
            td_error = rewards[t] + self.gamma * value_next[:,t] * (1 - dones[t]) - value_old[:,t]
            running_advantage = td_error + self.gamma * self.clip_value * (1 - dones[t]) * running_advantage
            advantages[t] = running_advantage
        return advantages

    def train(self, num_episodes):
        for episode in range(num_episodes):
            self.env.reset()
            state = self.env.obs
            done = False
            transitions = []
            total_reward = 0
            step = 0
            while not done and step < self.batch:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                transitions.append(Transition(state, action, reward, next_state, done))
                total_reward += reward
                state = next_state
                step += 1
            self.learn(transitions)
            print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}")    

barriers, player_x, player_y, box, goals = load_function_from_json('maps')
env = SokobanEnv(playerXY=(player_x, player_y) , barriers=barriers , boxes=box , goals=goals)  # Instantiate environment
env.reset()
ppo_agent = PPOAgent(env)  # Instantiate agent
ppo_agent.train(1000)  # Train agent for 1000 episodes

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        
        self.conv1 = nn.Conv2d(input_dim[0], 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv_out_size = self._get_conv_output_dim(self.input_dim)
        self.fc1 = nn.Linear(self.conv_out_size, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # Max pooling with 2x2 kernel, reduces input size by half
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        
        x = x.view(-1, self.conv_out_size)  # Keep batch size for batch mode
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(4,int(self.output_dim/4))
        # get the max of each row
        x = torch.max(x, 0)[0]
        return x

    def _get_conv_output_dim(self, shape):
        dummy_input = torch.zeros(1, *shape)
        dummy_output = self._forward_conv(dummy_input)
        return int(torch.prod(torch.tensor(dummy_output.shape)))

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        return x    

class DQNAgent:
    def __init__(self, env: SokobanEnv, replay_capacity=10000, batch_size=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.000_001 , strategy='linear'):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN((batch_size,10,10), len(env.action_space)*batch_size).to(self.device)
        self.target_net = DQN((batch_size,10,10), len(env.action_space)*batch_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        self.loss_history = []
        self.best_loss = 100
        
        self.win_history = self.load_win_history()
        self.remove_duplicates()
        
        # self.classifier = Classifier().to(self.device)
        # self.load_classifier('classifier')
        # self.train_classifier()

        self.discount_factor = 0.99
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step_number = 0
        
        self.strategy = strategy

        self.replay_memory = ReplayMemory(replay_capacity)
        self.batch_size = batch_size
    
    def remove_duplicates(self):
        win_unique = {}
        for win in self.win_history:
            attributes = [str(hash(str(key) + str(len(win[key])))) for key in win]
            uniq_id = hash(''.join(attributes))
            win_unique[uniq_id] = win
        self.win_history = list(win_unique.values())
        with open('win_history.pkl', 'wb') as f:
            pickle.dump(self.win_history, f)
        
    def choose_action(self, state):
        self.step_number += 1
        
        if self.strategy == 'linear':
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        elif self.strategy == 'exponential':
            self.epsilon = self.epsilon_start * math.exp(-1 * self.epsilon_decay * self.step_number)
        elif self.strategy == 'constant':
            self.epsilon = self.epsilon_end
        elif self.strategy == 'epsilon_greedy':
            self.epsilon = max(self.epsilon_end, self.epsilon_start - self.epsilon_decay * self.step_number)
            if np.random.rand() < self.epsilon:
                return np.random.choice(self.env.action_space)
        else:
            raise ValueError('Invalid strategy')
        
        with torch.no_grad():
            if np.random.rand() < self.epsilon:
                return np.random.choice(self.env.action_space)
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
                state_tensor = state_tensor.repeat(self.batch_size, 1 , 1)
                q_values = self.policy_net(state_tensor)
                # print(q_values)
                action = q_values.argmax().item()
                # print("max item size:", q_values[action])
                action = action % len(self.env.action_space)
                return action

    def learn(self):
        if len(self.replay_memory) < self.batch_size:
            return

        transitions = self.replay_memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(self.device)
        
        
        # Predict Q-values for the current state
        q_values = self.policy_net(state_batch)
        q_values = q_values.unsqueeze(0)
        # print(f'{q_values.shape=}')
        # print(f'{action_batch.shape=}')

        # Select the Q-value corresponding to the action
        q_values = q_values.gather(1, action_batch.unsqueeze(0))

        # Compute expected Q-values for the next state
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)
            next_q_values = next_q_values.unsqueeze(0)
            expected_q_values = reward_batch + self.discount_factor * next_q_values * (1 - done_batch)
            # done bach = 1024
            # reward bach = 1024
            # next_q_values = 1 * 1024

        # Calculate the loss and perform backpropagation
        loss = self.loss_fn(q_values, expected_q_values)
        self.loss_history.append(loss.item())
        if len(self.loss_history) > 500:
            # remove first 5000 logs
            print('#'*25)
            loss_calc = sum(self.loss_history)/len(self.loss_history)
            if loss_calc < self.best_loss:
                self.best_loss = loss_calc
                self.target_net.load_state_dict(self.policy_net.state_dict())
                torch.save(self.policy_net.state_dict(), 'DQN')
            print(f'Current loss :{loss}')
            print('#'*25)
            self.loss_history = self.loss_history[250:] 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def add_to_win_history(self, obs_history):
        self.win_history.append(obs_history)
        # pickle.dump(self.win_history, open('win_history.pkl', 'wb'))
        with open('win_history.pkl', 'wb') as f:
            pickle.dump(self.win_history, f)
        
    def load_win_history(self):
        try:
            with open('win_history.pkl', 'rb') as f:
                self.win_history = pickle.load(f)
            # self.win_history = pickle.load(open('win_history.pkl', 'rb'))
        except FileNotFoundError:
            self.win_history = []
        return self.win_history
    
    def find_win_moves(self, current_obs):
        obs_hash = hashlib.sha256(current_obs).hexdigest()
        solutions = [{'length': len(win), 'action': win[obs_hash]['action']} for win in self.win_history if obs_hash in win]
        if solutions:
            return min(solutions, key=lambda x: x['length'])['action']
        return None

    def train(self, num_episodes, reset_threshold=250_000):
        best_average_reward = -float('inf')
        best_episode = 0
        
        pygame.init()
        screen_width, screen_height = 800, 400
        screen = pygame.display.set_mode((screen_width, screen_height))

        for episode in range(num_episodes):
            self.env.reset()
            state = self.env.obs
            done = False
            total_reward = 0
            current_step = 0
            verbose = 1000
            moves = {}
            pause = False
            
            while not done:
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_p:
                            pause = not pause
                            print("Paused")
                        if event.key == pygame.K_r:
                            break
                
                       
                            
                if random.random() < PROBABILITY_OF_USING_HISTORY:
                    # start = time.time()
                    action_v1 = self.find_win_moves(state)
                    # print(f"Time taken to find win moves: {time.time() - start}")
                    if action_v1 is not None:
                        action = action_v1
                    else:
                        action = self.choose_action(state)
                else:
                    action = self.choose_action(state) 
                        
                
                if pause:
                    pygame.event.clear()  # Clear any previous events to avoid queuing up events
                    event = pygame.event.wait()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_p:
                            pause = not pause
                            print("Paused")
                        if event.key == pygame.K_r:
                            break
                        if event.key == pygame.K_LEFT:
                            action = 0
                        elif event.key == pygame.K_RIGHT:
                            action = 1
                        elif event.key == pygame.K_UP:
                            action = 2
                        elif event.key == pygame.K_DOWN:
                            action = 3
                        else:
                            action = 0       
            
                next_state, reward, done, extra_data = self.env.step(action)
                if extra_data['valid'] == True:
                    moves[hashlib.sha256(state).hexdigest()]={'action':action, 'obs' : state}
                
                if done:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    self.add_to_win_history(moves)
                    moves = {}
            
                self.replay_memory.push(Transition(state, action, reward, next_state, False))
                self.learn()
                total_reward += reward
                state = next_state

                if env.step_count - env.last_box_move > reset_threshold:
                    env.step_count = 0
                    env.last_box_move = 0
                    print("Resetting environment")
                    self.env.reset()
                    done = True
                
                # Log progress
                if current_step % verbose == 0 and current_step > 0:
                    print(f"Current step: {current_step}, Total reward: {total_reward}, Average reward: {total_reward / current_step}")
                    print(f"Epsilon : {self.epsilon}")
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                
                 
                """ arena is 10*10 grid
                player == 4
                box == 3 
                goal == 2
                barrier == 1
                empty space == 0
                """
                arena = self.env.arena
                obs = self.env.obs
                
                # render this arena
                block_size = screen_width // 20
                
                for i in range(10):
                    for j in range(10):
                        if arena[i][j] == 4:
                            pygame.draw.rect(screen, (255, 0, 0), (block_size * i, block_size * j, block_size, block_size))
                        elif arena[i][j] == 3:
                            pygame.draw.rect(screen, (0, 128, 0), (block_size * i, block_size * j, block_size, block_size))
                        elif arena[i][j] == 2:
                            pygame.draw.rect(screen, (0, 0, 255), (block_size * i, block_size * j, block_size, block_size))
                        elif arena[i][j] == 1:
                            pygame.draw.rect(screen, (255, 255, 255), (block_size * i, block_size * j, block_size, block_size))
                        else:
                            pygame.draw.rect(screen, (0, 0, 0), (block_size * i, block_size * j, block_size, block_size))
                            
                for i in range(10):
                    for j in range(10):
                        obs_val = obs[i][j]
                        if obs_val < -1:
                            color = (255, 0, 0)
                        else:
                            color = (min(int(25*((obs_val+1.5)*(obs_val+1.5))), 255), min(int(5*((obs_val+1.25)*(obs_val+1.25))), 255), min(int(25*((obs_val+1.125)*(obs_val+1.125))), 255))

                        # draw text on screen with value of obs
                        font = pygame.font.Font(None, 12)
                        text = font.render(str(round(obs_val, 2)), True, (128, 128, 255))
                        screen.blit(text, (block_size * i + 10, block_size * j + 10))
                        pygame.draw.rect(screen, color, (block_size * i + 10 * block_size, block_size * j, block_size, block_size))
                        
                pygame.display.flip()
                current_step += 1
                
            # Print episode information
            print(f'Epsilon :{self.epsilon}')
            print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}, Average Reward: {total_reward / current_step}")
            if total_reward / current_step > best_average_reward:
                best_average_reward = total_reward / current_step
                best_episode = episode + 1
            
        print(f"Best average reward achieved: {best_average_reward} at episode {best_episode}")

    def save_q_network(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_q_network(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        


# print(env.obs)
# agent = DQNAgent(env, replay_capacity=10_000 , batch_size=256, strategy='epsilon_greedy' , epsilon_start=0.025)  # Instantiate agent
# # agent.load_q_network('DQN')  # Load the NN weights from a file
# agent.train(num_episodes=2500 , reset_threshold=250)  # Train the agent
# agent.save_q_network('DQN')  # Save the NN weights to a file
