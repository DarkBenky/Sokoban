import random
import os
import numpy as np
import pickle
import json
import sys
import pygame

def load_function_from_json(folder_path = 'maps' , map_name = None , rnd = True , check_if_arena_exists = True):

    if check_if_arena_exists:
        memory = pickle.load(open('memory.pkl', 'rb'))

        run_data = []
        run_labels = []

        pointer = 0
        while pointer < len(memory):
            for i in range(pointer, len(memory)):
                if memory[i]['win']:
                    temp_s = [log['obs'] for log in memory[pointer:i+1]]
                    temp_l_s = [log['action'] for log in memory[pointer:i+1]]

                    pointer = i + 1

                    run_data.append(temp_s)
                    run_labels.append(temp_l_s)
                    break
        
    def contract_obs():
        obs = np.zeros((10,10,4), dtype=np.int8)
        encoding = np.int8(255)
        obs[player_x][player_y][0] = encoding
        for x, y in box:
            obs[x][y][1] = encoding
        for x, y in goals:
            obs[x][y][2] = encoding
        for x, y in barriers:
            obs[x][y][3] = encoding
        return obs
        
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
                obs = contract_obs()
                for run in run_data:
                    if not np.array_equal(obs, run[0]):
                        continue
                    else:
                        break
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
        """
        Take a step in the environment based on the selected action
        :param action: int, action index
        :return: np.array, new observation , bool, win status , bool, valid move
        """
        # Take a step in the environment based on the selected action
        direction = self.get_direction_from_action(action)

        success = self.move(direction)

        if success:
            self.obs = self.construct_obs()
            if self.is_win():
                return self.obs , True , True
            else:
                return self.obs , False , True
        return self.obs , False , False
    
    def reset(self):
        barriers, player_x, player_y, box, goals = load_function_from_json('maps')
        
        self.playerXY = np.array((player_x, player_y))
        self.barriers = np.array(barriers)
        self.boxes = np.array(box)
        self.goals = np.array(goals)
        self.obs = self.construct_obs()
        return self.obs
        
    def copy(self):
        return SokobanEnv(self.playerXY, self.barriers, self.boxes, self.goals)



import tensorflow as tf
import time


class EnvVisualizer():
    def __init__(self, ENV , model):
        self.model = model
        self.ENV = ENV
        self.WIDTH = 600
        self.HEIGHT = 600
        self.grid_size = min(self.WIDTH // ENV.obs.shape[1], self.HEIGHT // ENV.obs.shape[0])

        self.barrier_color = (128, 128, 128)
        self.player_color = (0, 128, 255)
        self.box_color = (255, 165, 0)
        self.goal_color = (255, 0, 0)
        self.empty_color = (255, 255, 255)
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Sokoban")
        self.current_step = 0
        self.max_seq_len = 64
        self.Seq_data = np.zeros((1, self.max_seq_len, 10 * 10 * 4))
        self.Seq_data[0][0] = ENV.obs.reshape(10 * 10 * 4)
        self.moves = []
        # load the sequential model
        self.model = tf.keras.models.load_model('models/LSTM-seq_32_0.5_5564.keras')
        
        
        self.load_memory()
        
    def fill_seq_data(self):
        temp_env = self.ENV.copy()
        actions = []
        print(self.Seq_data.shape)
        for i in range(self.current_step, self.max_seq_len - 1):
            predicted = self.model.predict(self.Seq_data)
            # Decode the one-hot encoded labels
            temp_env_seq = self.ENV.copy()
            temp_action = []
            for action in predicted[0]:
                current_action = np.argmax(action)
                if current_action == 4:
                    break
                temp_action.append(current_action)
                obs , win , _ = temp_env_seq.step(current_action)
                if win:
                    print('win')
                    return temp_action , True
            action = np.argmax(predicted[0][i])
            actions.append(action)
            obs, win , valid_move = temp_env.step(action)
            self.Seq_data[0][i + 1] = obs.reshape(10 * 10 * 4)
            if win:
                print('win')
                return actions , True
        return actions , False
     
     
    def add_to_seq_data(self, obs):
        if self.current_step < self.max_seq_len - 1:
            self.Seq_data[0][self.current_step] = obs.reshape(10 * 10 * 4)
            self.current_step += 1
        else:
            self.Seq_data[0][:self.max_seq_len // 2] = self.Seq_data[0][self.max_seq_len // 2:]
            self.Seq_data[0][self.max_seq_len // 2:] = np.zeros((self.max_seq_len // 2, 10 * 10 * 4))
            self.Seq_data[0][self.max_seq_len // 2] = obs.reshape(10 * 10 * 4)
            self.current_step = self.max_seq_len // 2 + 1
            

    def load_memory(self):
        try:
            self.memory = pickle.load(open('memory.pkl', 'rb'))
        except:
            self.memory = []
            
    def clear_to_win(self):
        length = len(self.memory) - 1
        found = False
        
        while found != True:
            if self.memory[length]['win'] == True:
                return
            self.memory.pop(length)
            length -= 1
            
                
    def save_memory_reset(self, obs , action , win):
        self.memory.append({
            'version': 0.1,
            'obs': obs,
            'action': action,
            'win' : win
        })
        if win:
            pickle.dump(self.memory, open('memory.pkl', 'wb'))
            self.ENV.reset()

    def draw_grid(self, labels):
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

        # Get player's position in pixels
        player_x_pixel = self.ENV.playerXY[1] * self.grid_size
        player_y_pixel = self.ENV.playerXY[0] * self.grid_size

        # Define label positions relative to the player
        label_up_pos = (player_x_pixel, player_y_pixel - self.grid_size)
        label_down_pos = (player_x_pixel, player_y_pixel + self.grid_size)
        label_left_pos = (player_x_pixel - self.grid_size, player_y_pixel)
        label_right_pos = (player_x_pixel + self.grid_size, player_y_pixel)

        # Write the probability of each action
        font = pygame.font.Font(None, 18)
        label_up = font.render(str(labels[0][0]), True, (0, 0, 0))
        label_down = font.render(str(labels[0][1]), True, (0, 0, 0))
        label_left = font.render(str(labels[0][2]), True, (0, 0, 0))
        label_right = font.render(str(labels[0][3]), True, (0, 0, 0))

        # Blit labels on squares around the player
        self.screen.blit(label_up, label_up_pos)
        self.screen.blit(label_down, label_down_pos)
        self.screen.blit(label_left, label_left_pos)
        self.screen.blit(label_right, label_right_pos)
        
        # Draw the heat map of the actions
        for i, label in enumerate(labels[0]):
            color = 255 * label
            color = (0, color, 0)
            if i == 0:
                pygame.draw.rect(self.screen, color, (player_x_pixel, player_y_pixel - self.grid_size, self.grid_size, self.grid_size), 3)
            elif i == 1:
                pygame.draw.rect(self.screen, color, (player_x_pixel, player_y_pixel + self.grid_size, self.grid_size, self.grid_size), 3)
            elif i == 2:
                pygame.draw.rect(self.screen, color, (player_x_pixel - self.grid_size, player_y_pixel, self.grid_size, self.grid_size), 3)
            elif i == 3:
                pygame.draw.rect(self.screen, color, (player_x_pixel + self.grid_size, player_y_pixel, self.grid_size, self.grid_size), 3)
        
        if len(self.moves) == 0:
            return
        
        # draw the moves
        box_x = player_x_pixel
        box_y = player_y_pixel
        
        color = 255 / 8
        for i, move in enumerate(self.moves):
            if i > 8:
                break
            if move == 0:
                box_y -= self.grid_size
            elif move == 1:
                box_y += self.grid_size
            elif move == 2:
                box_x -= self.grid_size
            elif move == 3:
                box_x += self.grid_size
            elif move == 4:
                # continue to the next sequence
                continue
                
            pygame.draw.rect(self.screen, (0 , color*i , color*i), (box_x, box_y, self.grid_size, self.grid_size), 3)
                

    def main(self , desable_ai = False):
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                        if event.key == pygame.K_UP:
                            action = 0
                        elif event.key == pygame.K_DOWN:
                            action = 1
                        elif event.key == pygame.K_LEFT:
                            action = 2
                        elif event.key == pygame.K_RIGHT:
                            action = 3

                        self.current_step += 1
                        obs, win , valid_move = self.ENV.step(action)
                        if not desable_ai:
                            self.add_to_seq_data(obs)
                            self.save_memory_reset(obs, action, win)
                            self.moves  , win_seq = self.fill_seq_data()
                        
                            if win_seq:
                                obs_seq = []
                                actions_seq = []
                                wins = []
                                for move in self.moves:
                                    obs , win_real , _= self.ENV.step(move)
                                    self.draw_grid([[0, 0, 0, 0, 0]])
                                    pygame.display.update()
                                    time.sleep(0.2)
                                    obs_seq.append(obs)
                                    actions_seq.append(move)
                                    wins.append(win_real)
                                    if win_real:
                                        for i in range(len(obs_seq)):
                                            self.save_memory_reset(obs_seq[i], actions_seq[i], wins[i])
                                        self.clear_to_win()
                                        self.ENV.reset()
                                        self.Seq_data = np.zeros((1, self.max_seq_len, 10 * 10 * 4))
                                        self.Seq_data[0][0] = self.ENV.obs.reshape(10 * 10 * 4)
                                        self.current_step = 0
                        else:
                            if valid_move:
                                self.save_memory_reset(obs, action, win)
                        
                    elif event.key == pygame.K_r:
                        self.clear_to_win()
                        self.ENV.reset()
                        self.Seq_data = np.zeros((1, self.max_seq_len, 10 * 10 * 4))
                        self.Seq_data[0][0] = self.ENV.obs.reshape(10 * 10 * 4)
                        self.current_step = 0
                   
            prepared_data = self.ENV.obs.reshape(1, 10, 10, 4)
            labels = model.predict(prepared_data) 

            self.screen.fill(self.empty_color)
            self.draw_grid(labels)
            pygame.display.update()
            clock.tick(60)


# from collections import deque

# class DQNAgent:
#     def __init__(self, state_shape, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
#         self.state_shape = state_shape
#         self.action_size = action_size
#         self.learning_rate = learning_rate
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay
#         self.model = self.build_model()
#         self.replay_buffer = deque(maxlen=10000)

#     def build_model(self):
#         model = tf.keras.Sequential([
#             tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.state_shape),
#             tf.keras.layers.MaxPooling2D((2, 2)),
#             tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#             tf.keras.layers.MaxPooling2D((2, 2)),
#             tf.keras.layers.Flatten(),
#             tf.keras.layers.Dense(128, activation='relu'),
#             tf.keras.layers.Dense(self.action_size, activation='linear')
#         ])
#         model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
#         return model

#     def remember(self, state, action, reward, next_state, done):
#         self.replay_buffer.append((state, action, reward, next_state, done))

#     def act(self, state):
#         if np.random.rand() <= self.epsilon:
#             return np.random.choice(self.action_size)
#         q_values = self.model.predict(state)
#         return np.argmax(q_values[0])

#     def replay(self, batch_size):
#         if len(self.replay_buffer) < batch_size:
#             return
#         minibatch = random.sample(self.replay_buffer, batch_size)
#         for state, action, reward, next_state, done in minibatch:
#             target = reward
#             if not done:
#                 target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
#             target_f = self.model.predict(state)
#             target_f[0][action] = target
#             self.model.fit(state, target_f, epochs=1, verbose=1)
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay


# # Instantiate the environment and agent
# barriers, player_x, player_y, box, goals = load_function_from_json('maps')
# env = SokobanEnv((player_x, player_y), barriers, box, goals)
# state_shape = env.obs.shape
# action_size = 4
# agent = DQNAgent(state_shape, action_size)
# EPISODES = 10_000
# MAX_MOVES = 256
# BATCH_SIZE = 256

# # Training loop
# for episode in range(EPISODES):
#     state = env.reset()
#     state = np.reshape(state, [1, state_shape[0], state_shape[1], state_shape[2]])
#     total_reward = 0
#     done = False
#     iteration = 0
#     while not done and iteration < MAX_MOVES:
#         action = agent.act(state)
#         next_state, done , valid_move = env.step(action)
#         if done:
#             reward = 10
#             print('win')
#         else:
#             reward = -0.1
#         if valid_move:
#             reward = 0.1
#         else:
#             reward = -0.5
#         next_state = np.reshape(next_state, [1, state_shape[0], state_shape[1], state_shape[2]])
#         agent.remember(state, action, reward, next_state, done)
#         state = next_state
#         total_reward += reward
#         if done:
#             print("episode: {}/{}, score: {}".format(episode, EPISODES, total_reward))
#             break
#         agent.replay(BATCH_SIZE)





from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    'best_sokoban_model.keras',      
    monitor='val_accuracy',       
    verbose=1,                     
    save_best_only=True,          
    mode='max',                    
    save_weights_only=False      
)


model = Sequential()

# Add convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(10, 10, 4)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

# Add dense layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))

# Output layer for multi-label classification
model.add(Dense(4, activation='softmax'))  # 4 output labels


# train_data, train_labels, test_data, test_labels = load_data()

# input_shape = train_data[0].shape
# num_classes = train_labels[0].shape[0]

# model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(
#     train_data, 
#     train_labels, 
#     epochs=100, 
#     batch_size=512, 
#     validation_data=(test_data, test_labels),
#     callbacks=[checkpoint]
# )

# # Evaluate the model
# test_loss, test_acc = model.evaluate(test_data, test_labels)
# print("Test accuracy:", test_acc)

# load the best model
model.load_weights('best_sokoban_model.keras')

barriers, player_x, player_y, box, goals = load_function_from_json('maps')
env = SokobanEnv((player_x, player_y), barriers, box, goals)
visualizer = EnvVisualizer(env , model)

visualizer.main(True)