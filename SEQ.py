import numpy as np
import tensorboard as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape , Dropout
import random
import pickle


def load_data(split = 0.8):
    with open('memory.pkl', 'rb') as f:
        memory = pickle.load(f)
    
    labels = []
    data = []
    
    for log in memory:
        temp_labels = [0, 0, 0, 0]
        temp_labels[log['action']] = 1
        labels.append(temp_labels)
        data.append(log['obs'])
    
    data = np.array(data)
    labels = np.array(labels)
    
    return data[:int(len(data)*split)], labels[:int(len(data)*split)], data[int(len(data)*split):], labels[int(len(data)*split):]

def whole_sequences(split=0.8, max_seq_len=64):
    with open('memory.pkl', 'rb') as f:
        memory = pickle.load(f)
    
    data_seq = []
    label_seq = []
    
    pointer = 0
    while pointer < len(memory):
        for i in range(pointer, len(memory)):
            if memory[i]['win']:
                temp_s = [log['obs'] for log in memory[pointer:i+1]]
                temp_l_s = [log['action'] for log in memory[pointer:i+1]]
                
                temp_s = np.array(temp_s)
                temp_l_s = np.array(temp_l_s)
                
                # Pad sequences to the same length
                if len(temp_s) < max_seq_len:
                    temp_s = np.concatenate([temp_s, np.zeros((max_seq_len - len(temp_s), 10, 10, 4), dtype=np.int8)])
                    # Pad the labels with the last action with number 4
                    temp_l_s = np.concatenate([temp_l_s, np.full((max_seq_len - len(temp_l_s)), 4, dtype=np.int8)])
                
                # Reshape the input data to be compatible with LSTM
                temp_s = temp_s.reshape(max_seq_len, 10*10*4)
                
                # One-hot encode the labels
                temp_l_s = np.eye(5)[temp_l_s]
                
                data_seq.append(temp_s)
                label_seq.append(temp_l_s)
                
                pointer = i + 1
                break
        
    data_seq = np.array(data_seq)
    label_seq = np.array(label_seq)
            
    return data_seq[:int(len(data_seq)*split)], label_seq[:int(len(data_seq)*split)], data_seq[int(len(data_seq)*split):], label_seq[int(len(data_seq)*split):]


max_seq_len = 64

train_seq, train_labels_seq, test_seq, test_labels_seq = whole_sequences(0.8, max_seq_len)

print(f"{train_seq.shape=}")
print(f"{train_labels_seq.shape=}")

# Define a TensorBoard callback for logging
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

# Define the model name
checkpoint_path = 'LSTM-seq' + str(random.randint(0 , 10000)) + '.keras'

# Define a ModelCheckpoint callback to save the best model
checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')

# Define the RNN model
model = Sequential([
    LSTM(256, input_shape=(max_seq_len, 10*10*4), return_sequences=True),
    Dropout(0.5),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dense(256, activation='relu'),
    Dense(train_labels_seq.shape[1] * train_labels_seq.shape[2], activation='softmax'),
    Reshape((max_seq_len, train_labels_seq.shape[2]))
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with checkpoint saving and TensorBoard logging
model.fit(train_seq, train_labels_seq, epochs=500, batch_size=512, 
          callbacks=[checkpoint_callback, tensorboard_callback], validation_split=0.2)

# Load the best saved model
model.load_weights(checkpoint_path)

# Evaluate the best model
test_data = np.array(test_seq)
test_labels = np.array(test_labels_seq)

test_loss, test_acc = model.evaluate(test_data, test_labels)
print("Test accuracy:", test_acc)