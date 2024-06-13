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
                        data.append({'obs': s, 'action': l})
                    else:
                        data.append({'obs': s, 'action': l})
    return data
                    
print(load_seq(8))



    


            
