import random
import os
import numpy as np
import json
import sys
import pickle
import hashlib
import time
# import time
# from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import pprint
import pygame
# from numba import jit

INVALID_MOVE_PENALTY = -2
PREFORMED_MOVE_PENALTY = -0.5
STACKED_BOX = -10
WIN_REWARD = 100
BOX_PUSH_REWARD = 1.5
BOX_CLOSE_TO_GOAL_REWARD = 2.5
PLAYER_CLOSE_TO_BOX_REWARD = 0.5
TARGET_UPDATE = 10
LEARNING_RATE = 0.01
BOX_NEAR_BARRIER_PENALTY = 0.125
PROBABILITY_OF_USING_HISTORY = 0.95

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

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, output_dim)  # Output dimension matches the number of actions

    def forward(self, x):
        # Flatten the input tensor to be compatible with fully connected layers
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(10 * 10, 512)  # Input size: 10x10, Output size: 64
        self.fc2 = nn.Linear(512, 256)        # Input size: 64, Output size: 32
        self.fc3 = nn.Linear(256, 4)         # Input size: 32, Output size: 4
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 10 * 10)  # Flatten the input to a vector
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)  # Apply softmax activation along dimension 1 (the features dimension)
        return x

class DQNAgent:
    def __init__(self, env: SokobanEnv, replay_capacity=10000, batch_size=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.000_001 , strategy='linear'):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(np.prod(env.obs.shape), len(env.action_space)).to(self.device)
        self.target_net = DQN(np.prod(env.obs.shape), len(env.action_space)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        
        self.win_history = self.load_win_history()
        self.remove_duplicates()
        
        self.classifier = Classifier().to(self.device)
        self.train_classifier()

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

    def train_classifier(self, episodes=3):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.01)
        
        training_data = []
        training_labels = []
        
        for win in self.win_history:
            for obs_hash, data in win.items():
                # Check if data has key 'obs'
                if 'obs' in data:
                    obs = data['obs']
                    a = data['action']
                    action = np.zeros(4)
                    action[a] = 1
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                    action_tensor = torch.tensor(action, dtype=torch.float32).to(self.device)
                    
                    training_data.append(obs_tensor)
                    training_labels.append(action_tensor)
                    
        for episode in range(episodes):
            total_loss = 0
            for i in range(len(training_data)):
                optimizer.zero_grad()
                output = self.classifier(training_data[i])
                loss = F.binary_cross_entropy_with_logits(output, action_tensor.unsqueeze(0))
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
            print(f'Episode: {episode}, Loss: {total_loss / len(training_data)}\r ')
        
        # save model 
        torch.save(self.classifier.state_dict(), 'classifier')
        
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
        elif self.strategy == 'boltzmann':
            if self.epsilon_decay < 0:
                raise ValueError("Temperature (epsilon_decay) must be positive for Boltzmann exploration")
            q_values = self.policy_net(torch.tensor(state, dtype=torch.float32).to(self.device))
            boltzmann_probs = F.softmax(q_values / self.epsilon_decay, dim=-1)
            action = torch.multinomial(boltzmann_probs, 1)
            return action.item()
        elif self.strategy == 'softmax':
            if self.epsilon_decay < 0:
                raise ValueError("Temperature (epsilon_decay) must be positive for Softmax exploration")
            q_values = self.policy_net(torch.tensor(state, dtype=torch.float32).to(self.device))
            softmax_probs = F.softmax(q_values / self.epsilon_decay, dim=-1)
            action = torch.multinomial(softmax_probs, 1)
            return action.item()
        else:
            raise ValueError('Invalid strategy')
        
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
            q_values = self.policy_net(state_tensor)
            if np.random.rand() < self.epsilon:
                return np.random.choice(self.env.action_space)
            else:
                return q_values.argmax().item()  # Return the action with the highest Q-value

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

        q_values = self.policy_net(state_batch)
        q_values = q_values.gather(1, action_batch.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            expected_q_values = reward_batch + self.discount_factor * next_q_values * (1 - done_batch)

        loss = self.loss_fn(q_values, expected_q_values)

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
            moves_v2 = []

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
                
                if pause == False:
                    action = self.choose_action(state)
                else:
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
                if random.random() < PROBABILITY_OF_USING_HISTORY:
                    # action_v2 = self.history_replay.find(state)
                    # if action_v2 is not None:
                    #     action = action
                    #     print('using V2')
                    # else:
                    # print('using V1')
                    action_v1 = self.find_win_moves(state)
                    if action_v1 is not None:
                        action = action_v1
            
                next_state, reward, done, extra_data = self.env.step(action)
                if extra_data['valid'] == True:
                    moves[hashlib.sha256(state).hexdigest()]={'action':action, 'obs' : state}
                
                if done:
                    self.add_to_win_history(moves)
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    moves = {}
            
                self.replay_memory.push(Transition(state, action, reward, next_state, done))
                self.learn()
                total_reward += reward
                state = next_state

                if env.step_count - env.last_box_move > reset_threshold:
                    env.step_count = 0
                    env.last_box_move = 0
                    print("Resetting environment")
                    self.env.reset()
                    self.target_net.load_state_dict(self.policy_net.state_dict())
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

        print(f"Best average reward achieved: {best_average_reward} at episode {best_episode}")

    def save_q_network(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_q_network(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        

barriers, player_x, player_y, box, goals = load_function_from_json('maps')
env = SokobanEnv(playerXY=(player_x, player_y) , barriers=barriers , boxes=box , goals=goals)  # Instantiate environment
env.reset()
# print(env.obs)
agent = DQNAgent(env, replay_capacity=1_000_000 , batch_size=256, strategy='epsilon_greedy' , epsilon_start=0.05)  # Instantiate agent
agent.train(num_episodes=2_500 , reset_threshold=200)  # Train the agent
agent.save_q_network('DQN')  # Save the NN weights to a file
