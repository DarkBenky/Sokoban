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
        map  = [['.' for _ in range(self.map_size[0])] for _ in range(self.map_size[1])]
        map[self.y][self.x] = 'X'
        for cord in self.box_cords:
            map[cord[1]][cord[0]] = 'O'
        for cord in self.wall_cords:
            map[cord[1]][cord[0]] = '#'
        for cord in self.final_cords:
            map[cord[1]][cord[0]] = '$'
        if show == 'human':
            return '\n'.join([' '.join(row) for row in map])
        return map
        
    
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
                   
        
              
    
    def return_map_3d_array(self):
        
        # path = self.find_path_to_goal()
        
        # path_map = [[0 for _ in range(self.map_size[0])] for _ in range(self.map_size[1])]
        # for cord in path:
        #     path_map[cord[1]][cord[0]] = 1
        
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
            
        if complex:
            return [full_map , player_map, box_map, wall_map, final_map , distance_box_final , distance_box_player , distance_final_player]
        return full_map
    
    def find_path_to_goal(self):
        def heuristic(node, goal):
            return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

        def get_neighbors(node):
            x, y = node
            return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

        start = self.box_cords[0]
        goal = self.final_cords[0]

        open_set = PriorityQueue()
        open_set.put((0, start, []))  # (priority, node, path)

        closed_set = set()

        while not open_set.empty():
            _, current, path = open_set.get()

            if current == goal:
                return path

            if current in closed_set:
                continue
            
            if len(path) > 50:
                return []

            closed_set.add(current)

            for neighbor in get_neighbors(current):
                if neighbor not in self.wall_cords and neighbor not in closed_set:
                    new_path = path + [current]
                    priority = len(new_path) + heuristic(neighbor, goal)
                    open_set.put((priority, neighbor, new_path))

        return []  # No path found
        
    
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

def load_function_from_json(folder_path = 'maps'):
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    if not files:
        raise FileNotFoundError("No JSON files found in the specified folder.")

    # Select a random JSON file
    random_file = random.choice(files)
    file_path = os.path.join(folder_path, random_file)

    # Load data from the selected file
    data = json.load(open(file_path))
    
    # Extract information from the data
    barriers = data.get('barriers', [])
    player_x, player_y = data.get('player', [0, 0])
    box = data.get('box', [])
    goals = data.get('goals', [])

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
    def __init__(self , map_size = (20, 20) , reset_step = 2500 , max_invalid_move_reset = 100 , config_dict = {} , logging = True):
        self.game = Game(player_x=3, player_y=6, player_char=1, wall_char=2, empty_char=0, box_char=3, final_char=4 , map_size=map_size)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=4, shape=(8, 10, 10 ), dtype=np.float32)
        self.config_dict = config_dict
        self.logging = logging
        self.map_size = map_size
        self.current_step = 0
        self.reset_step = reset_step
        self.max_invalid_move_reset = max_invalid_move_reset

    def step(self, action):
        action = ['up', 'down', 'left', 'right'][action]
        
        old_player_x , old_player_y = self.game.x , self.game.y
        old_x, old_y = self.game.box_cords[0]
        # distance_box_final , distance_box_player , distance_final_player = self.game.generate_heatmap()
        
        valid_move = self.game.move(action)
        
        self.current_step += 1
        
        if self.current_step > self.reset_step or self.max_invalid_move_reset == 0:
            return self.reset(), 0, True, {}
        
        if not valid_move:
            self.max_invalid_move_reset -= 1
            if self.current_step % 2 == 0 and self.logging:
                wandb.log({"reward": self.config_dict["invalid_move_reward"],
                        'valid_move': 0,
                        "box_final_distance": self.game.calculate_distance(self.game.box_cords[0][0], self.game.box_cords[0][1], self.game.final_cords[0][0], self.game.final_cords[0][1]),
                        "box_player_distance": self.game.calculate_distance(self.game.box_cords[0][0], self.game.box_cords[0][1], self.game.x, self.game.y),
                        "final_player_distance": self.game.calculate_distance(self.game.final_cords[0][0], self.game.final_cords[0][1], self.game.x, self.game.y),
                        })
            return self.game.return_map_3d_array(), self.config_dict["invalid_move_reward"], False, {}
        
        if self.game.check_win():
            if self.logging:
                wandb.log({
                    "reward": self.config_dict["win_reward"],
                    'valid_move': 1,
                    'box_final_distance': 0,
                })
            print('win')
            return self.reset(), self.config_dict["win_reward"], True, {}
        
        if valid_move:
            
            reward = self.config_dict["preform_step"]
            new_x, new_y = self.game.box_cords[0]
            
            if new_x != old_x or new_y != old_y:
                reward += self.config_dict["box_move_reward"]
                
            new_player_x , new_player_y = self.game.x , self.game.y
            
            old_distance = self.game.calculate_distance(old_player_x, old_player_y, self.game.box_cords[0][0], self.game.box_cords[0][1])
            new_distance = self.game.calculate_distance(new_player_x, new_player_y, self.game.box_cords[0][0], self.game.box_cords[0][1])
                
            if old_distance > new_distance:
                reward += self.config_dict["box_player_reward"]
            
            old_distance = self.game.calculate_distance(old_x, old_y, self.game.final_cords[0][0], self.game.final_cords[0][1])
            new_distance = self.game.calculate_distance(new_x, new_y, self.game.final_cords[0][0], self.game.final_cords[0][1])
            
            if old_distance > new_distance:
                reward += self.config_dict["box_goal_reward"]
                
            if new_distance < 1.5:
                reward += self.config_dict["box_near_goal"]
                
            if new_distance < 3:
                reward += self.config_dict["box_close_goal"]
                
            old_distance = self.game.calculate_distance(old_player_x, old_player_y, self.game.final_cords[0][0], self.game.final_cords[0][1])
            new_distance = self.game.calculate_distance(new_player_x, new_player_y, self.game.final_cords[0][0], self.game.final_cords[0][1])
            
            if old_distance > new_distance:
                reward += self.config_dict["final_player_reward"]
                
            if self.current_step % 20 == 0 and self.logging:
                wandb.log({ "reward": reward,
                            'valid_move': 1,
                            "box_final_distance": self.game.calculate_distance(new_x, new_y, self.game.final_cords[0][0], self.game.final_cords[0][1]),
                            "box_player_distance": self.game.calculate_distance(new_x, new_y, self.game.x, self.game.y),
                            "final_player_distance": self.game.calculate_distance(self.game.final_cords[0][0], self.game.final_cords[0][1], self.game.x, self.game.y),
                        })
            return self.game.return_map_3d_array(), reward, False, {}


    def reset(self):
        walls , player_x , player_y , box , goals = load_function_from_json()
        self.current_step = 0
        self.max_invalid_move_reset = self.config_dict["max_invalid_move_reset"]

        self.game = Game(player_x=player_x, player_y=player_y, player_char=1, wall_char=2, empty_char=0, box_char=3, final_char=4, wall_cords=walls, final_cords=[goals], box_cords=[box] , map_size=self.map_size, path_char=5)
        new_map = self.game.return_map_3d_array()
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
                steps = 75
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
        'net_arch': {'pi': [random.randint(64, 1024) for _ in range(random.randint(1, 5))],
                     'vf': [random.randint(64, 1024) for _ in range(random.randint(1, 5))]},
        'net_arch_dqn': [random.randint(64, 1024) for _ in range(random.randint(1, 5))],
        'batch_size': random.choice([32, 64, 128, 256]),
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
        'invalid_move_reward': random.uniform(-20.0, 0.0),
        'max_invalid_move_reset': random.randint(5, 20),
        'model_type': random.choice(['PPO', 'DQN']),
        'policy': random.choice(['MlpPolicy']),
        'folder_path_for_models': 'models-tone',
    }
            


def toneParams(tries = 100):
    best_params = {}
    best_result = -np.inf
    best_model = None
    for _ in range(tries):
        config_dict = generate_random_params()
        wandb.init(project="sokoban-tone", config=config_dict , name=config_dict['model_name'])
        env = Monitor(Env(config_dict['map_size'], config_dict['reset'], config_dict['max_invalid_move_reset']))
        env.reset()
        if config_dict['model_type'] == 'PPO':
            model = PPO("MlpPolicy", env, verbose=1 , learning_rate=config_dict['learning_rate'], policy_kwargs=dict(net_arch=config_dict['net_arch']) , batch_size=config_dict['batch_size'])
        elif config_dict['model_type'] == 'DQN':
            if config_dict['policy'] == 'CnnPolicy':
                env = Monitor(Env(config_dict['map_size'], config_dict['reset'], config_dict['max_invalid_move_reset'] , complex=False))
                env.reset()
                model = DQN("CnnPolicy", env, verbose=1 , learning_rate=config_dict['learning_rate'], policy_kwargs=dict(net_arch=config_dict['net_arch_dqn']) , batch_size=config_dict['batch_size'])
            else:
                env = Monitor(Env(config_dict['map_size'], config_dict['reset'], config_dict['max_invalid_move_reset']))
                env.reset()
                model = DQN("MlpPolicy", env, verbose=1 , learning_rate=config_dict['learning_rate'], policy_kwargs=dict(net_arch=config_dict['net_arch_dqn']) , batch_size=config_dict['batch_size'])
        model.learn(total_timesteps=100_000 , progress_bar=True , callback=CustomCallback(eval_freq=2_500 , config_dict=config_dict))
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
# best_params , best_model = toneParams()
# best_model.save(best_params['folder_path_for_models'] + '/' + best_params['model_name']+ '-best-tone')
# file_name = best_params['folder_path_for_models'] + '/' + best_params['model_name']+'-best-tone'+'.json'
# with open(file_name, 'w') as f:
#     json.dump(best_params, f)


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


def test_model(model_path='models/model-1709509845-best.zip', config_file='models/model-1709509845-best.json'):
    with open(config_file) as f:
        config_dict = json.load(f)
    env = Monitor(Env(config_dict['map_size'], config_dict['reset'], config_dict['max_invalid_move_reset'] , config_dict , logging=False))
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
        steps = 75
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
                    maps.append('win')
        model_data.append(maps)
    print(f"Model performance: {model_performance}" , f"Win count: {(win_count / 10) * 100}%")
    for data in model_data:
        if data[-1] == 'win':
            for i in range(len(data)):
                print(data[i][0])
    return model_performance , model_data

test_model()
    
   
