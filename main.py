import gym
import wandb
from stable_baselines3 import PPO , DQN
import torch
import torch.nn as nn
import random
import os
from gym import spaces
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from queue import PriorityQueue
import json
from path import pathfinder

class Game():
    def __init__(self, player_x, player_y, player_char='X', wall_char='#', empty_char='.', box_char='O', final_char='$', final_cords=[(3, 3)],
                 box_cords=[(5, 6)], wall_cords = [(0, 0),(0, 9),(9, 0),(9, 9)], map_size=(10, 10) , path_char = 'P'):
        self.x = player_x
        self.y = player_y
        self.player_char = player_char
        self.wall_char = wall_char
        self.empty_char = empty_char
        self.final_char = final_char
        self.box_char = box_char
        self.path_char = path_char
        self.box_cords = box_cords
        self.final_cords = final_cords
        self.wall_cords = wall_cords
        self.map_size = map_size
        self.full_map = []
        self.full_map_path = []

    def move(self, direction) -> bool:
        def convert_direction(direction):
            if direction == 'up' or direction == 'w' or direction == 'W':
                return (0, -1)
            if direction == 'down' or direction == 's' or direction == 'S':
                return (0, 1)
            if direction == 'left' or direction == 'a' or direction == 'A':
                return (-1, 0)
            if direction == 'right' or direction == 'd' or direction == 'D':
                return (1, 0)

        def is_valid_move(x, y):
            return 0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]

        new_x = self.x + convert_direction(direction)[0]
        new_y = self.y + convert_direction(direction)[1]

        def check_wall_collision(x, y):
            return [x, y] in self.wall_cords or (x, y) in self.wall_cords
        def check_box_collision(x, y):
            return [x, y] in self.box_cords or (x, y) in self.box_cords
    
        if is_valid_move(new_x, new_y):
            if check_wall_collision(new_x, new_y) == False and check_box_collision(new_x, new_y) == False:
                self.x = new_x
                self.y = new_y
                return True
            elif check_box_collision(new_x, new_y):
                for i, cord in enumerate(self.box_cords):
                    if cord == [new_x, new_y] or cord == (new_x, new_y):
                        if not check_wall_collision(convert_direction(direction)[0] + new_x, convert_direction(direction)[1] + new_y) and not check_box_collision(convert_direction(direction)[0] + new_x, convert_direction(direction)[1] + new_y) and  is_valid_move(convert_direction(direction)[0] + new_x, convert_direction(direction)[1] + new_y):
                            self.box_cords[i] = (convert_direction(direction)[0] + new_x, convert_direction(direction)[1] + new_y)
                            self.x = new_x
                            self.y = new_y
                            return True
                        else:
                            return False
            else:
                return False
        return False

    def __str__(self):
        map  = [[self.empty_char for _ in range(self.map_size[0])] for _ in range(self.map_size[1])]
        map[self.y][self.x] = self.player_char
        for cord in self.box_cords:
            map[cord[1]][cord[0]] = self.box_char
        for cord in self.wall_cords:
            map[cord[1]][cord[0]] = self.wall_char
        for cord in self.final_cords:
            map[cord[1]][cord[0]] = self.final_char
        return '\n'.join([' '.join(row) for row in map])
    
    def show_map(self, show = 'human'):
        if show == 'human':
            map_  = [['.' for _ in range(self.map_size[0])] for _ in range(self.map_size[1])]
            map_[self.x][self.y] = 'X'
            for cord in self.box_cords:
                map_[cord[1]][cord[0]] = 'O'
            for cord in self.wall_cords:
                map_[cord[1]][cord[0]] = '#'
            for cord in self.final_cords:
                map_[cord[1]][cord[0]] = '$'
                return '\n'.join([' '.join(row) for row in map_])
        elif show == 'array':
            map_  = [[0 for _ in range(self.map_size[0])] for _ in range(self.map_size[1])]
            map_[self.x][self.y] = 1
            for cord in self.box_cords:
                map_[cord[1]][cord[0]] = 2
            for cord in self.wall_cords:
                map_[cord[1]][cord[0]] = 3
            for cord in self.final_cords:
                map_[cord[1]][cord[0]] = 4
            return map_
        elif show == 'combined':
            map_  = [['.' for _ in range(self.map_size[0])] for _ in range(self.map_size[1])]
            map_[self.x][self.y] = 'X'
            for cord in self.box_cords:
                map_[cord[1]][cord[0]] = 'O'
            for cord in self.wall_cords:
                map_[cord[1]][cord[0]] = '#'
            for cord in self.final_cords:
                map_[cord[1]][cord[0]] = '$'
                return map_
        else:
            raise ValueError("Invalid value for show parameter. Expected 'human' or 'array'.")
        
    
    def __repr__(self):
        return self.__str__()
    
    def calculate_distance(self, start_x: int, start_y: int, end_x: int, end_y: int):
        if start_x == end_x and start_y == end_y:
            return 0
        
        dx = end_x - start_x
        dy = end_y - start_y
        distance = (dx**2 + dy**2)**0.5
        return distance

    
    def generate_heatmap(self):
        array_2d = [[0 for _ in range(self.map_size[0])] for _ in range(self.map_size[1])]
        box_x , box_y = self.box_cords[0] 
        final_x , final_y = self.final_cords[0]
        player_x , player_y = self.x, self.y
        
        distance_box = array_2d.copy()
        distance_player = array_2d.copy()
        distance_final = array_2d.copy()
        
        for i in range(len(array_2d)):
            for j in range(len(array_2d)):
                distance = self.calculate_distance(box_x, box_y, i, j)
                distance_box[j][i] = 1/(distance + 0.1)
                distance = self.calculate_distance(player_x, player_y, i , j)
                distance_player[j][i] = 1/(distance + 0.1)
                distance = self.calculate_distance(final_x, final_y, i , j)
                distance_final[j][i] = 1/(distance + 0.1)
                
        for cord  in self.wall_cords:
            distance_box[cord[0]][cord[1]] = 0
            distance_player[cord[0]][cord[1]] = 0
            distance_final[cord[0]][cord[1]] = 0
            
        box_player = distance_player.copy()
        box_final = distance_final.copy()
        final_player = distance_player.copy()
        
        for i in range(len(array_2d)):
            for j in range(len(array_2d)):
                box_player[j][i] = (distance_box[j][i] + distance_player[j][i]) / 2
                box_final[j][i] = (distance_box[j][i] + distance_final[j][i]) / 2
                final_player[j][i] = (distance_final[j][i] + distance_player[j][i]) /2
        return box_player , box_final , final_player 
        # return box_player , box_final , final_player , distance_box , distance_player , distance_final
                   
        
              
    
    def return_map_3d_array(self , complex_map = True):
        distance_box_final , distance_box_player , distance_final_player = self.generate_heatmap()
            
        player_map = [[0 for _ in range(self.map_size[0])] for _ in range(self.map_size[1])]
        full_map = [[0 for _ in range(self.map_size[0])] for _ in range(self.map_size[1])]
        player_map[self.y][self.x] = self.player_char
        full_map[self.y][self.x] = 1

        box_map = [[0 for _ in range(self.map_size[0])] for _ in range(self.map_size[1])]
        for cord in self.box_cords:
            box_map[cord[1]][cord[0]] = self.box_char
            full_map[cord[1]][cord[0]] = 2
            
        wall_map = [[0 for _ in range(self.map_size[0])] for _ in range(self.map_size[1])]
        for cord in self.wall_cords:
            wall_map[cord[1]][cord[0]] = self.wall_char
            full_map[cord[1]][cord[0]] = 3
        
        final_map = [[0 for _ in range(self.map_size[0])] for _ in range(self.map_size[1])] 
        for cord in self.final_cords:
            final_map[cord[1]][cord[0]] = self.final_char
            full_map[cord[1]][cord[0]] = 4
        
        if complex_map: 
            try:
                path = pathfinder(current_x=self.box_cords[0][0], current_y=self.box_cords[0][1], visited=[(self.box_cords[0][0], self.box_cords[0][1])], finish_x=self.final_cords[0][0], finish_y=self.final_cords[0][1], width=self.map_size[0], height=self.map_size[1], matrix=full_map)
                full_map_path = full_map.copy()
                for cord in path:
                    full_map_path[cord[0]][cord[1]] = 5
            except:
                full_map_path = full_map
            self.full_map_path = full_map_path
            
        self.full_map = full_map
        
        if complex_map:
            return [full_map , full_map_path , player_map, box_map, wall_map, distance_box_final , distance_box_player , distance_final_player]
        return full_map
    
    
    def check_win(self):
        for box_cord in self.box_cords:
            box_x , box_y = box_cord
            for final_cord in self.final_cords:
                final_x , final_y = final_cord
                if box_x != final_x or box_y != final_y:
                    return False
        return True
        # box_x , box_y = self.box_cords[0]
        # final_x , final_y = self.final_cords[0]
        # if box_x == final_x and box_y == final_y:
        #     return True
        # return False

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
    
    print(f"Loaded map from {file_path}")

    return barriers, player_x, player_y, box, goals


# walls, player_x , player_y, box, goals = load_function_from_json('map.json')
# game = Game(player_x=player_x, player_y=player_y, player_char='X', wall_char='#', empty_char='.', box_char='O',final_char='$', final_cords=[goals], wall_cords=walls, box_cords=[box] , map_size=(10, 10))

# while not game.check_win():
#     try:
#         print(game.move(input()))
#     except Exception as e:
#         print('not valid move')
#     print(game)
    
class Env(gym.Env):
    def __init__(self , map_size = (20, 20) , reset_step = 2500 , logging = True , complex_map = False , reset_after_invalid_move = 10):
        self.game = Game(player_x=3, player_y=6, player_char=1, wall_char=2, empty_char=0, box_char=3, final_char=4 , map_size=map_size)
        self.action_space = spaces.Discrete(4)
        if complex_map:
            self.observation_space = spaces.Box(low=0, high=4, shape=(8, 10, 10 ), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0, high=4, shape=(1, 10, 10 ), dtype=np.float32)
        self.logging = logging
        self.map_size = map_size
        self.current_step = 0
        self.reset_step = reset_step
        self.complex_map = complex_map
        self.reset_after_invalid_move = reset_after_invalid_move

    def step(self, action):
        action = ['up', 'down', 'left', 'right'][action]  # Convert the action index to a corresponding action string
        
        old_player_x, old_player_y = self.game.x, self.game.y  # Store the current player position
        if config_dict['complex_map']:
            path_old = self.game.full_map_path
        old_x, old_y = self.game.box_cords[0]  # Store the current box position
        
        valid_move = self.game.move(action)  # Perform the move and check if it is a valid move
        
        self.current_step += 1  # Increment the current step counter
        
        if self.current_step > self.reset_step or self.reset_after_invalid_move <= 0:
            return self.reset(), config_dict['no_win_reward'], True, {}  # If the maximum number of steps is reached, reset the environment and return a no-win reward
        
        if not valid_move:
            self.reset_after_invalid_move -= 1  # Decrement the number of invalid moves left before resetting the environment
            if self.logging:
                wandb.log({
                    "reward": config_dict["invalid_move_reward"],
                    'valid_move': 0,
                    'box_final_distance': self.game.calculate_distance(self.game.box_cords[0][0], self.game.box_cords[0][1], self.game.final_cords[0][0], self.game.final_cords[0][1]),
                })  # Log the reward, valid move flag, and box-final distance if the move is invalid
            return  self.game.return_map_3d_array(self.complex_map), config_dict["invalid_move_reward"], False, {}  # Reset the environment and return an invalid move reward
        
        if self.game.check_win():
            if self.logging:
                wandb.log({
                    "reward": config_dict["win_reward"],
                    'valid_move': 1,
                    'box_final_distance': 0,
                })  # Log the reward, valid move flag, and box-final distance if the game is won
            print('win')
            return self.reset(), config_dict["win_reward"], True, {}  # Reset the environment and return a win reward
        
        if valid_move:
            reward = config_dict["preform_step"]
            new_x, new_y = self.game.box_cords[0]
            
            if path_old[new_x][new_y] == 5 and config_dict['complex_map']:
                reward += config_dict["box_moved_on_correct_path_reward"]
            
            # If the box has moved, add the box move reward to the total reward.
            if new_x != old_x or new_y != old_y:
                reward += config_dict["box_move_reward"]
                
            new_player_x, new_player_y = self.game.x, self.game.y
            
            # Calculate the distances before and after the move for the box and the player.
            old_distance = self.game.calculate_distance(old_player_x, old_player_y, self.game.box_cords[0][0], self.game.box_cords[0][1])
            new_distance = self.game.calculate_distance(new_player_x, new_player_y, self.game.box_cords[0][0], self.game.box_cords[0][1])
                
            # If the player has moved closer to the box, add the box player reward to the total reward.
            if old_distance > new_distance:
                reward += config_dict["box_player_reward"]
            
            old_distance = self.game.calculate_distance(old_x, old_y, self.game.final_cords[0][0], self.game.final_cords[0][1])
            new_distance = self.game.calculate_distance(new_x, new_y, self.game.final_cords[0][0], self.game.final_cords[0][1])
            
            # If the box has moved closer to the goal, add the box goal reward to the total reward.
            if old_distance > new_distance:
                reward += config_dict["box_goal_reward"]
                
            # If the box is near the goal, add the box near goal reward to the total reward.
            if new_distance < 1.5:
                reward += config_dict["box_near_goal"]
                
            # If the box is close to the goal, add the box close goal reward to the total reward.
            if new_distance < 3:
                reward += config_dict["box_close_goal"]
                
            old_distance = self.game.calculate_distance(old_player_x, old_player_y, self.game.final_cords[0][0], self.game.final_cords[0][1])
            new_distance = self.game.calculate_distance(new_player_x, new_player_y, self.game.final_cords[0][0], self.game.final_cords[0][1])
            
            # If the player has moved closer to the goal, add the final player reward to the total reward.
            if old_distance > new_distance:
                reward += config_dict["final_player_reward"]
                
            if self.current_step % 2 == 0 and self.logging:
                wandb.log({ "reward": reward,
                            'valid_move': 1,
                            "box_final_distance": self.game.calculate_distance(new_x, new_y, self.game.final_cords[0][0], self.game.final_cords[0][1]),
                        })
            return self.game.return_map_3d_array(self.complex_map), reward, False, {}


    def reset(self):
        walls , player_x , player_y , box , goals = load_function_from_json()
        self.current_step = 0
        self.reset_after_invalid_move = config_dict['max_invalid_move_reset']
        self.game = Game(player_x=player_x, player_y=player_y, player_char=1, wall_char=2, empty_char=0, box_char=3, final_char=4, wall_cords=walls, final_cords=[goals], box_cords=[box] , map_size=self.map_size, path_char=5)
        new_map = self.game.return_map_3d_array(self.complex_map)
        return np.array(new_map)

    def render(self , mode='human'):
        return self.game.show_map(show=mode) , self.game.box_cords[0] , self.game.final_cords[0]
    
      
class CustomCallback(BaseCallback):
    def __init__(self, verbose=0 , eval_freq = 1000 , config_dict = {}):
        super(CustomCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.config = config_dict
        self.best_result = -np.inf
    
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
            model_performance = 0
            for _ in range(10):
                obs = self.model.env.reset()
                done = False
                steps = 100
                while not done and steps > 0:
                    action, _states = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.model.env.step(action)
                    steps -= 1
                    if done:
                        if reward >= 50:
                            model_performance += 100 / (steps + 1)
            if model_performance > self.best_result:
                self.config['model_performance'] = model_performance
                self.best_result = model_performance
                self.model.save(self.config['folder_path_for_models']+ '/' +self.config['model_name'] + '-best')
                file_name = self.config['folder_path_for_models'] + '/' + self.config['model_name']+ '-best' + '.json'
                with open(file_name, 'w') as f:
                    json.dump(self.config, f)
                wandb.log({'model_performance': model_performance})
            print(f"Model performance: {model_performance}")
        return True
              
        
def generate_model_name():
    import time
    return f"model-{int(time.time())}"

def generate_random_params():
    return {
        'learning_rate': random.uniform(0.0001, 0.1),
        'net_arch': {'pi': [random.randint(256, 2048) for _ in range(random.randint(2, 5))],
                     'vf': [random.randint(256, 2048) for _ in range(random.randint(2, 5))]},
        'net_arch_dqn': [random.randint(1024, 4096) for _ in range(random.randint(2, 5))],
        'batch_size': random.choice([32, 64, 128, 256, 512, 1024, 2048]),
        'model_name': generate_model_name(),
        'map_size': (10, 10),
        'reset': random.randint(50, 200),
        'box_near_goal': random.uniform(0.0, 1.0),
        'box_close_goal': random.uniform(0.0, 1.0),
        'box_move_reward': random.uniform(-1.0, 1.0),
        'box_goal_reward': random.uniform(-1.0, 1.0),
        'box_player_reward': random.uniform(-1.0, 1.0),
        'final_player_reward': random.uniform(-1.0, 1.0),
        'preform_step': random.uniform(-1.0, 1.0),
        'win_reward': random.randint(50, 200),
        'invalid_move_reward': random.uniform(-10.0, 0.0),
        'no_win_reward': random.uniform(-2.5, 0.0),
        'model_type': random.choice(['PPO', 'DQN']),
        'policy': random.choice(['MlpPolicy']),
        'folder_path_for_models': 'models-tone-complex-2',
        'complex_map': random.choice([True, False]),
        'max_invalid_move_reset': random.uniform(5.0, 20.0),
        'box_moved_on_correct_path_reward': random.uniform(0, 1.0)
    }
            


def toneParams(tries = 100):
    best_params = {}
    best_result = -np.inf
    best_model = None
    for _ in range(tries):
        global config_dict
        config_dict = generate_random_params()
        wandb.init(project="sokoban-tone-complex", config=config_dict , name=config_dict['model_name'])
        
        if config_dict['model_type'] == 'PPO':
            env = Monitor(Env(config_dict['map_size'], config_dict['reset'], logging=True , complex_map=config_dict['complex_map'], reset_after_invalid_move=config_dict['max_invalid_move_reset']))
            env.reset()
            model = PPO("MlpPolicy", env, verbose=1 , learning_rate=config_dict['learning_rate'], policy_kwargs=dict(net_arch=config_dict['net_arch']) , batch_size=config_dict['batch_size'])
        elif config_dict['model_type'] == 'DQN':
            if config_dict['policy'] == 'CnnPolicy':
                env = Monitor(Env(config_dict['map_size'], config_dict['reset'], logging=True , complex_map=config_dict['complex_map'] , reset_after_invalid_move=config_dict['max_invalid_move_reset']))
                env.reset()
                model = DQN("CnnPolicy", env, verbose=1 , learning_rate=config_dict['learning_rate'], policy_kwargs=dict(net_arch=config_dict['net_arch_dqn']) , batch_size=config_dict['batch_size'])
            else:
                env = Monitor(Env(config_dict['map_size'], config_dict['reset'], logging=True , complex_map=config_dict['complex_map'] , reset_after_invalid_move=config_dict['max_invalid_move_reset']))
                env.reset()
                model = DQN("MlpPolicy", env, verbose=1 , learning_rate=config_dict['learning_rate'], policy_kwargs=dict(net_arch=config_dict['net_arch_dqn']) , batch_size=config_dict['batch_size'])
        model.learn(total_timesteps=150_000 , progress_bar=True , callback=CustomCallback(eval_freq=2_500 , config_dict=config_dict))
        model_performance = 0
        for _ in range(10):
            obs = model.env.reset()
            done = False
            steps = 75
            while not done and steps > 0:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = model.env.step(action)
                steps -= 1
                if done:
                    if reward >= 50:
                        model_performance += 100 / (steps + 1)
        if model_performance > best_result:
            best_result = model_performance
            config_dict['model_performance'] = model_performance
            best_params = config_dict
            best_model = model
        wandb.finish()
    return best_params , best_model



# NOTE: uncomment the following line to run the toneParams function
best_params , best_model = toneParams()
best_model.save(best_params['folder_path_for_models'] + '/' + best_params['model_name']+ '-best-tone')
file_name = best_params['folder_path_for_models'] + '/' + best_params['model_name']+'-best-tone'+'.json'
with open(file_name, 'w') as f:
    json.dump(best_params, f)


# config_dict = {
#     'learning_rate': 0.01,
#     'net_arch': {'pi': [512,512], 'vf': [1024,1024,512,512]},
#     'net_arch_dqn': [1024, 1024, 1024, 512],
#     'batch_size': 70,
#     'model_name': generate_model_name(),
#     'map_size': (10, 10),
#     'reset': 116,
#     'box_near_goal': 0.69,
#     'box_close_goal' : 0.56,
#     'box_move_reward': -0.01,
#     'box_goal_reward': -0.10,
#     'box_player_reward': -0.2,
#     'final_player_reward': -0.02,
#     'preform_step' : -0.17,
#     'win_reward': 145.1,
#     'invalid_move_reward': -9.69,
#     'max_invalid_move_reset': 8.2,
#     'model_type': 'DQN',
#     'policy': 'MlpPolicy',
#     'folder_path_for_models': 'models',
# }

# wandb.init(project="sokoban-0.1", config=config_dict , name=config_dict['model_name'])   


# if config_dict['model_type'] == 'PPO':
#     env = Monitor(Env(config_dict['map_size'], config_dict['reset'], config_dict['max_invalid_move_reset']))
#     env.reset()
#     model = PPO("MlpPolicy", env, verbose=1 , learning_rate=config_dict['learning_rate'], policy_kwargs=dict(net_arch=config_dict['net_arch']) , batch_size=config_dict['batch_size'])
# elif config_dict['model_type'] == 'DQN':
#     if config_dict['policy'] == 'CnnPolicy':
#         env = Monitor(Env(config_dict['map_size'], config_dict['reset'], config_dict['max_invalid_move_reset'] , complex=False))
#         env.reset()
#         model = DQN("CnnPolicy", env, verbose=1 , learning_rate=config_dict['learning_rate'], policy_kwargs=dict(net_arch=config_dict['net_arch_dqn']) , batch_size=config_dict['batch_size'])
#     # NOTE: Maybe try different policy types but for CnnPolicy we need to change the observation space
#     # TODO: try more experiments with CnnPolicy and study more about it
#     else:
#         env = Monitor(Env(config_dict['map_size'], config_dict['reset'], config_dict['max_invalid_move_reset']))
#         env.reset()
#         model = DQN("MlpPolicy", env, verbose=1 , learning_rate=config_dict['learning_rate'], policy_kwargs=dict(net_arch=config_dict['net_arch_dqn']) , batch_size=config_dict['batch_size'])
#     # NOTE: DQN models are probably better
#     # TODO: try more experiments with DQN models
    
# model.learn(total_timesteps=5_000_000 , progress_bar=True , callback=CustomCallback(eval_freq=2_500 , config_dict=config_dict))


def test_model(model_path='models-tone-complex-2/model-1709678562-best.zip', config_file='models-tone-complex-2/model-1709678562-best.json'):
    with open(config_file) as f:
        global config_dict
        config_dict = json.load(f)
    env = Monitor(Env(config_dict['map_size'], config_dict['reset'], logging=False , complex_map=config_dict['complex_map'], reset_after_invalid_move=config_dict['max_invalid_move_reset']))
    if config_dict['model_type'] == 'PPO':
        model = PPO.load(model_path)
        model.set_env(env)
    elif config_dict['model_type'] == 'DQN':
        model = DQN.load(model_path)
        model.set_env(env)
    model_performance = 0
    model_data = []
    win_count = 0
    for i in range(10):
        obs = model.env.reset()
        maps = []
        done = False
        steps = 100
        while not done and steps > 0:
            action, _states = model.predict(obs, deterministic=True)
            # print(model.env.render())
            maps.append(model.env.render())
            obs, reward, done, info = model.env.step(action)
            steps -= 1
            if done:
                if reward >= 50:
                    model_performance += 100 / (steps + 1)
                    win_count += 1
                    maps.append(['win', [] , []])
        model_data.append(maps)
    print(f"Model performance: {model_performance}" , f"Win count: {(win_count / 10) * 100}%")
    for data in model_data:
            for i in range(len(data)):
                print(data[i][0])
                print("box position: ",data[i][1])
                print("final position: ",data[i][2])
    return model_performance , model_data

# test_model()
    
   
