import random
import os
import numpy as np
import pickle
import json
import sys
import pygame

def load_function_from_json(folder_path = 'maps' , map_name = None , rnd = True):
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

    # update the player position to be random
    if rnd:
        done = False
        while done != True:
            player_x = random.randint(0, 9)
            player_y = random.randint(0, 9)
            if [player_x, player_y] not in box and [player_x, player_y] not in barriers:
                done = True
    
    return barriers, player_x, player_y, box, goals


class SokobanEnv:
    def __init__(self, playerXY=(0, 0), barriers=[], boxes=[], goals=[]):
        self.playerXY = np.array(playerXY)
        self.boxes = np.array(boxes)
        self.goals = np.array(goals)
        self.barriers = np.array(barriers)
        self.obs = self.construct_obs()

    def construct_obs(self):
        # represent the state of the environment as a 2D grid with each color channel representing a different object
        obs = np.zeros((10,10,4), dtype=np.int8)
        encoding = np.int8(255)
        obs[self.playerXY[0], self.playerXY[1], 0] = encoding
        obs[self.boxes[:, 0], self.boxes[:, 1], 1] = encoding
        obs[self.goals[:, 0], self.goals[:, 1], 2] = encoding
        obs[self.barriers[:, 0], self.barriers[:, 1], 3] = encoding
        return obs


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
        return (0 <= position[0] < self.obs.shape[0] and
                0 <= position[1] < self.obs.shape[1] and
                self.obs[position[0]][position[1]][3] == 0)

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

        success = self.move(direction)

        if success:
            self.obs = self.construct_obs()
            if self.is_win():
                return self.obs , True
        return self.obs , False
    
    def reset(self):
        barriers, player_x, player_y, box, goals = load_function_from_json('maps')
        
        self.playerXY = np.array((player_x, player_y))
        self.barriers = np.array(barriers)
        self.boxes = np.array(box)
        self.goals = np.array(goals)
        self.obs = self.construct_obs()





class EnvVisualizer():
    def __init__(self, ENV):
        self.ENV = ENV
        self.WIDTH = 200
        self.HEIGHT = 200
        self.grid_size = min(self.WIDTH // ENV.obs.shape[1], self.HEIGHT // ENV.obs.shape[0])

        self.barrier_color = (128, 128, 128)
        self.player_color = (0, 128, 255)
        self.box_color = (255, 165, 0)
        self.goal_color = (255, 0, 0)
        self.empty_color = (255, 255, 255)
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Sokoban")

        self.load_memory()

    def load_memory(self):
        try:
            self.memory = pickle.load(open('memory.pkl', 'rb'))
        except:
            self.memory = []

    def save_memory_reset(self, obs , action , win):
        self.memory.append({
            'obs': obs,
            'action': action,
            'win' : win
        })
        if win:
            pickle.dump(self.memory, open('memory.pkl', 'wb'))
            self.ENV.reset()

    def draw_grid(self):
        for x, row in enumerate(self.ENV.obs):
            for y, cell in enumerate(row):
                if cell[0] != 0:
                    color = self.player_color
                elif cell[1] != 0:
                    color = self.box_color
                elif cell[2] != 0:
                    color = self.goal_color
                elif cell[3] != 0:
                    color = self.barrier_color
                else:
                    color = self.empty_color
                pygame.draw.rect(self.screen, color, (y * self.grid_size, x * self.grid_size, self.grid_size, self.grid_size))

    def main(self):
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        obs , win = self.ENV.step(0)
                        self.save_memory_reset(obs , 0 , win)
                    elif event.key == pygame.K_DOWN:
                        obs , win = self.ENV.step(1)
                        self.save_memory_reset(obs , 1 , win)
                    elif event.key == pygame.K_LEFT:
                        obs , win = self.ENV.step(2)
                        self.save_memory_reset(obs , 2 , win)
                    elif event.key == pygame.K_RIGHT:
                        obs , win = self.ENV.step(3)
                        self.save_memory_reset(obs , 3 , win)
                    elif event.key == pygame.K_r:
                        self.ENV.reset()
                   
                    

            self.screen.fill(self.empty_color)
            self.draw_grid()
            pygame.display.update()
            clock.tick(60)

barriers, player_x, player_y, box, goals = load_function_from_json('maps')
env = SokobanEnv((player_x, player_y), barriers, box, goals)
visualizer = EnvVisualizer(env)

visualizer.main()
    