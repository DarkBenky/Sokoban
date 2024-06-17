import numpy as np
import pickle
import random

def padding(data, max_seq_len):
    if len(data) < max_seq_len:
        data = np.concatenate([data, np.zeros((max_seq_len - len(data), 10, 10, 4), dtype=np.int8)])
    return data

def get_slices(arr, labels ,  size):
    if size <= 0 or size > len(arr):
        return [] , []

    slices = []
    l = []
    for i in range(len(arr) - size):
        slices.append(arr[i:i + size])
        l.append(labels[i + size])
    
    return slices , l

def load_seq(length=8):
    with open('memory.pkl', 'rb') as f:
        memory = pickle.load(f)
    
    labels = []
    obs = []

    pointer = 0
    while pointer < len(memory):
    # for i in range(len(memory)):
        temp_obs = [memory[pointer]['obs']]
        temp_act = [memory[pointer]['action']]
        for j in range(pointer + 1,len(memory)):
            if memory[j]['win']:
                labels.append(temp_act)
                obs.append(temp_obs)
                pointer = j + 1
                break
            else:
                temp_obs.append(memory[j]['obs'])
                temp_act.append(memory[j]['action'])


    data = []

    for label , action in zip(labels, obs):
        for i in range(1,length):
            slices , actions = get_slices(action , label , i)
            if len(slices) != 0:
                for s , l in zip(slices, actions):
                    if len(s) < length:
                        s = padding(s, length)
                        s = s.reshape(length, 10*10*4)
                        # One-hot encode the labels
                        l = np.eye(5)[l]
                        data.append({'obs': s, 'action': l})
                    else:
                        s = s.reshape(length, 10*10*4)
                        # One-hot encode the labels
                        l = np.eye(5)[l]
                        data.append({'obs': s, 'action': l})
    return data
                    
# print(load_seq(8))

def remove_duplicate_from_array(data: list):
    labels = []
    samples = set()
    unique_data = []

    for log in data:
        obs = log['obs']
        action = log['action']
        
        # Flatten and convert to tuple to make it hashable
        obs_flatten = tuple(obs.flatten())
        
        if obs_flatten not in samples:
            samples.add(obs_flatten)
            labels.append(action)
            unique_data.append(obs)

    return unique_data, labels

# data , labels = remove_duplicate_from_array(load_seq(8))


    


            
