import gym
# import wandb
from stable_baselines3 import PPO , DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
# import torch
# import torch.nn as nn
import random
import os
# from gym import spaces
import numpy as np
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.callbacks import BaseCallback
# from queue import PriorityQueue
import json
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from stable_baselines3.common.buffers import ReplayBuffer



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


class SokobanEnv(gym.Env):
    def __init__(self, playerXY=(0, 0), barriers=[], boxes=[], goals=[]):
        super(SokobanEnv, self).__init__()
        
        self.playerXY = np.array(playerXY)
        self.boxes = np.array(boxes)
        self.goals = np.array(goals)
        self.barriers = np.array(barriers)
        self.arena = self.construct_arena()
        
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=4, shape=self.arena.shape, dtype=np.int32)

    def construct_arena(self):
        # TODO: make it dynamic
        # if len(self.barriers) > 0:
        #     max_x = max(self.playerXY[0], *self.boxes[:, 0], *self.goals[:, 0], *self.barriers[:, 0]) + 1
        #     max_y = max(self.playerXY[1], *self.boxes[:, 1], *self.goals[:, 1], *self.barriers[:, 1]) + 1
        # else:
        #     max_x = max(self.playerXY[0], *self.boxes[:, 0], *self.goals[:, 0]) + 1
        #     max_y = max(self.playerXY[1], *self.boxes[:, 1], *self.goals[:, 1]) + 1

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
        
    def average_distance_of_boxes_to_goals(self):
        min_distances = []
        for box in self.boxes:
            distances = []
            for goal in self.goals:
                distance = np.linalg.norm(box - goal)
                distances.append(distance)
            min_distances.append(min(distances))
        return np.mean(min_distances)
    
    def render(self, mode='human' , close = False):
        for i in range(self.arena.shape[0]):
            for j in range(self.arena.shape[1]):
                current_position = (i, j)
                if np.array_equal(self.playerXY, current_position):
                    print('P', end=' ')
                elif any(np.array_equal(box, current_position) for box in self.boxes):
                    print('B', end=' ')
                elif any(np.array_equal(goal, current_position) for goal in self.goals):
                    print('G', end=' ')
                elif self.arena[i][j] == 1:
                    print('#', end=' ')
                else:
                    print('.', end=' ')
            print()
    
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
    
    def step(self, action):
        # Take a step in the environment based on the selected action
        direction = self.get_direction_from_action(action)

        previous_average_distance = self.average_distance_of_boxes_to_goals()
        
        success = self.move(direction)

        reward = 0.0
        done = False

        if success:
            current_average_distance = self.average_distance_of_boxes_to_goals()
            
            self.arena = self.construct_arena()
            if self.is_win():
                reward = 20.0
                # print("Win")
                done = True
            else:
                if current_average_distance < previous_average_distance:
                    reward = 0.5  # reward for reducing the average distance
                elif current_average_distance > previous_average_distance:
                    reward = -0.1  # Penalty for increasing the average distance
                else:
                    reward = -0.25  # Small negative reward for each step  

        else:
            reward = -1.0

        return self.arena, reward, done, {}
    
    def reset(self):
        barriers, player_x, player_y, box, goals = load_function_from_json('maps')
        
        self.playerXY = np.array((player_x, player_y))
        self.barriers = np.array(barriers)
        self.boxes = np.array(box)
        self.goals = np.array(goals)
        self.arena = self.construct_arena()

        return self.arena
    

class SaveBestModelCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq: int = 10000, save_path: str = 'models/best_model' , episodes = 10 , max_steps = 1000):
        super(SaveBestModelCallback, self).__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.episodes = episodes
        self.max_steps = max_steps
        self.best_mean_reward = -np.inf  # Initialize with negative infinity

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
            # Evaluate the model and update the best mean reward
            wins = 0
            mean_step_reward = 0.0
            illegal_moves = 0
            for _ in range(self.episodes):
                obs = self.eval_env.reset()
                done = False
                episode_reward = 0.0
                steps = self.max_steps

                while not done and steps > 0:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _ = self.eval_env.step(action)
                    episode_reward += reward
                    steps -= 1
                    
                    if reward <= -0.5:
                        illegal_moves += 1
                    
                    if done:
                        wins += 1
                    
                mean_step_reward += episode_reward / self.max_steps
            
            mean_reward = mean_step_reward / self.episodes

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                print(f"Saving new best model with mean step reward {mean_reward} to {self.save_path}")
                self.model.save(self.save_path)
                with open(self.save_path+".md", 'w') as f:
                    f.write("mean step reward :"+str(mean_reward)+"\n")
                    f.write("win rate :"+str(wins/self.episodes*100)+"%"+"\n")
                    f.write("illegal moves :"+str(illegal_moves/(self.episodes * self.max_steps) *100)+"%"+"\n")

# Create Sokoban environment
barriers, player_x, player_y, box, goals = load_function_from_json('maps')
env = SokobanEnv(playerXY=(player_x, player_y), barriers=barriers, boxes=box, goals=goals)
env.reset()

# Wrap the environment
env = DummyVecEnv([lambda: env])

# Set a higher entropy coefficient (adjust the value as needed)
ent_coef = 0.025

model = PPO("MlpPolicy", env, verbose=1 , ent_coef = ent_coef , n_epochs=64)

# Create a Sokoban environment for evaluation
eval_env = SokobanEnv(playerXY=(player_x, player_y), barriers=barriers, boxes=box, goals=goals)
eval_env.reset()
eval_env = DummyVecEnv([lambda: eval_env])

# Instantiate the SaveBestModelCallback
name = 'sokoban'+str(random.randint(0, 1000))
save_best_model_callback = SaveBestModelCallback(eval_env, eval_freq=10_000, save_path=f'models/best_model-{name}', episodes=10, max_steps=1000)

# Train the model
model.learn(total_timesteps=250_000 , progress_bar=True , log_interval=10 , callback=save_best_model_callback)

# # Save the trained model
# model.save("models/ppo_sokoban")

# # Evaluate the trained model
# mean_reward_list = []
# max_episode_steps = 1000
# tests = 25
# wins = 0

# for _ in range(tests):  # 10 evaluation episodes
#     obs = env.reset()
#     done = False
#     total_reward = 0
#     step_count = 0

#     while not done:
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, done, _ = env.step(action)
#         env.render()
#         total_reward += reward
#         step_count += 1

#         if done and reward == 1.0:
#             wins += 1
#             print(f"Win in {step_count} steps")

#         if step_count >= max_episode_steps:
#             done = True

#     mean_reward_list.append(total_reward)

# mean_reward = sum(mean_reward_list) / len(mean_reward_list)
# print(f"Mean reward: {mean_reward}")
# print(f"Win rate: {wins / tests * 100:.2f}%")

