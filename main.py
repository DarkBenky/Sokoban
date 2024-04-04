import random
import os
import numpy as np
import json
import sys
# import time
# from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import pygame
# from numba import jit

INVALID_MOVE_PENALTY = -5
PREFORMED_MOVE_PENALTY = -0.5
STACKED_BOX = -10
WIN_REWARD = 100
BOX_PUSH_REWARD = 1.5
BOX_CLOSE_TO_GOAL_REWARD = 2.5
PLAYER_CLOSE_TO_BOX_REWARD = 0.5
TARGET_UPDATE = 10
LEARNING_RATE = 0.001

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
        
        self.action_space = np.array([0, 1, 2, 3])
        
        self.running = False

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
    
    def box_is_movable(self,):
        corner_positions = [(0, 0), (9, 9), (0, 9), (9, 0)]
        if np.any(np.all(np.isin(self.boxes, corner_positions), axis=1)):
            return False
        for box in self.boxes:
            corner_list = [ ((1,0),(0,1)), ((1,0),(0,-1)),
                           ((-1,0),(0,1)), ((-1,0),(0,-1)) ]

            for corner in corner_list:
                corner1_dx, corner1_dy = corner[0]
                corner2_dx, corner2_dy = corner[1]
                corner1_x, corner1_y = box[0] + corner1_dx, box[1] + corner1_dy
                corner2_x, corner2_y = box[0] + corner2_dx, box[1] + corner2_dy
                
                if (corner1_x >= 0 and corner1_x < 10) and (corner1_y >= 0 and corner1_y < 10) and (corner2_x >= 0 and corner2_x < 10) and (corner2_y >= 0 and corner2_y < 10):
                    try:    
                        if self.arena[corner1_x][corner1_y] == 1 and self.arena[corner2_x][corner2_y] == 1:
                            return False
                    except IndexError:
                        return False
                else:
                    return False
        return True
    
    # @staticmethod
    # @jit(nopython=True)
    def get_direction_to_dx_dy(self, direction):
        if direction == 'W':
            return -1, 0
        elif direction == 'S':
            return 1, 0
        elif direction == 'A':
            return 0, -1
        elif direction == 'D':
            return 0, 1
    
        
    def is_box_close_to_goal(self):
        for box in self.boxes:
            for goal in self.goals:
                if np.linalg.norm(box - goal) == 1:
                    # Check if there's an empty space next to the goal
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        next_pos = goal + np.array([dx, dy])
                        if np.all(next_pos == box):
                            return True
        return False

    # @staticmethod
    # @jit(nopython=True)
    def move(self, direction , execute = True):
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

    # @staticmethod
    # @jit(nopython=True)
    def is_valid_move(self, position):
        return (0 <= position[0] < self.arena.shape[0] and
                0 <= position[1] < self.arena.shape[1] and
                self.arena[position[0]][position[1]] != 1)

    # @staticmethod
    # @jit(nopython=True)
    def is_win(self):
        return all(np.any(np.all(self.boxes == goal, axis=1) for goal in self.goals))
    
    # @staticmethod
    # @jit(nopython=True)
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
    
    def render(self):
        pass
    
    # @staticmethod
    # @jit(nopython=True)
    def valid_moves(self):
        possible_moves = ['W', 'S', 'A', 'D']
        valid_moves = []

        for move in possible_moves:
            if self.move(move , execute = False):
                valid_moves.append(move)
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
    
    # @staticmethod
    # @jit(nopython=True)
    def box_closer_to_goal(self, previous_box_positions, current_box_positions):
        for box in current_box_positions:
            if not any(np.array_equal(box, prev_box) for prev_box in previous_box_positions):
                # calculate the distance from the box to the goal
                for goal in self.goals:
                    distance = np.linalg.norm(box - goal)
                    if distance < 3:
                        return True
        return False
    
    def player_closer_to_box(self, previous_player_position, current_player_position):
        for box in self.boxes:
            if not np.array_equal(previous_player_position, current_player_position):
                # calculate the distance from the box to the goal
                distance = np.linalg.norm(current_player_position - box)
                if distance < 3:
                    return True
        return False
    
    def reverse_direction(self, direction):
        if direction == 'W':
            return 'S'
        elif direction == 'S':
            return 'W'
        elif direction == 'A':
            return 'D'
        elif direction == 'D':
            return 'A'
    
    def step(self, action):
        # Take a step in the environment based on the selected action
        direction = self.get_direction_from_action(action)

        checkpoint = self.is_box_close_to_goal()

        if checkpoint:
            if self.move(direction):
                if self.is_win():
                    return self.reset() , WIN_REWARD , True, {}
                else:
                    if self.is_box_close_to_goal():
                        self.arena = self.construct_arena()
                        return self.arena , BOX_CLOSE_TO_GOAL_REWARD, False, {}
                    else:
                        direction = self.reverse_direction(direction)
                        self.move(direction)
                        self.arena = self.construct_arena()
                        return self.arena , BOX_CLOSE_TO_GOAL_REWARD, False, {}
            else:
                return self.arena, INVALID_MOVE_PENALTY, False, {}
        else:
            box_positions = self.boxes.copy()
            previous_player_position = self.playerXY.copy()
            success = self.move(direction)

            done = False
            reward = 0

            if success:
                self.arena = self.construct_arena()
                if self.is_win():
                    reward = WIN_REWARD
                    print('WIN')
                    done = True
                else:
                    if self.box_is_movable() == False:
                        return self.reset(), STACKED_BOX, True, {}

                    reward += PREFORMED_MOVE_PENALTY
                    if np.any(self.boxes != box_positions):
                        reward += BOX_PUSH_REWARD
                        if self.box_closer_to_goal(box_positions , self.boxes):
                            reward += BOX_CLOSE_TO_GOAL_REWARD
                    if self.player_closer_to_box(previous_player_position , self.playerXY):
                        reward += PLAYER_CLOSE_TO_BOX_REWARD
            else:
                reward = INVALID_MOVE_PENALTY

            return self.arena, reward, done, {}
    
    def reset(self):
        barriers, player_x, player_y, box, goals = load_function_from_json('maps')
        
        self.playerXY = np.array((player_x, player_y))
        self.barriers = np.array(barriers)
        self.boxes = np.array(box)
        self.goals = np.array(goals)
        self.arena = self.construct_arena()

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
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, output_dim)  # Output dimension matches the number of actions

    def forward(self, x):
        # Flatten the input tensor to be compatible with fully connected layers
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class DQNAgent:
    def __init__(self, env, replay_capacity=10000, batch_size=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.000_001 , strategy='linear'):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(np.prod(env.arena.shape), len(env.action_space)).to(self.device)
        self.target_net = DQN(np.prod(env.arena.shape), len(env.action_space)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

        self.discount_factor = 0.99
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step_number = 0
        
        self.strategy = strategy

        self.replay_memory = ReplayMemory(replay_capacity)
        self.batch_size = batch_size

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

    def train(self, num_episodes, reset_threshold=250_000):
        best_average_reward = -float('inf')
        best_episode = 0
        
        pygame.init()
        screen_width, screen_height = 800, 800
        screen = pygame.display.set_mode((screen_width, screen_height))


        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            current_step = 0
            verbose = 1_000

            while not done:
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_memory.push(Transition(state, action, reward, next_state, done))
                self.learn()
                total_reward += reward
                state = next_state
                current_step += 1

                # Check for lack of progress
                if current_step % reset_threshold == 0:
                    if total_reward / current_step >= best_average_reward:
                        best_average_reward = total_reward / current_step
                    else:
                        print(f"Resetting at episode {episode}, Total reward: {total_reward}, Best Average reward: {best_average_reward}")
                        self.env.reset()
                
                # Log progress
                if current_step % verbose == 0 and current_step > 0:
                    print(f"Current step: {current_step}, Total reward: {total_reward}, Average reward: {total_reward / current_step}")
                    print(f"Epsilon : {self.epsilon}")

                """ arena is 10*10 grid
                player == 4
                box == 3 
                goal == 2
                barrier == 1
                empty space == 0
                """
                arena = self.env.arena
                
                # render this arena
                block_size = screen_width // 10
                
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
                            
                pygame.display.flip()  # Update display


            # Update target network periodically
            if episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

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
agent = DQNAgent(env, replay_capacity=10_000_000 , batch_size=10_000, strategy='epsilon_greedy')  # Instantiate agent
agent.train(num_episodes=1000)  # Train the agent
agent.save_q_table('DQN')  # Save the NN weights to a file
