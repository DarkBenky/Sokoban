import numpy as np
import tensorboard as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape, Dropout
import random
import pickle

def remove_duplicate_from_array(data: np.array, labels: np.array):
    data_list = data.tolist()
    labels_list = labels.tolist()
    
    unique_data = []
    unique_labels = []
    
    for i, item in enumerate(data_list):
        if item not in unique_data:
            unique_data.append(item)
            unique_labels.append(labels_list[i])
    
    return np.array(unique_data), np.array(unique_labels)

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
    data_seq, label_seq = remove_duplicate_from_array(data_seq, label_seq)
            
    return data_seq[:int(len(data_seq)*split)], label_seq[:int(len(data_seq)*split)], data_seq[int(len(data_seq)*split):], label_seq[int(len(data_seq)*split):]

def train_and_evaluate_model(architecture, params, train_seq, train_labels_seq, test_seq, test_labels_seq):
    # Define a TensorBoard callback for logging
    tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

    # Define the model name
    checkpoint_path = f"models/LSTM-seq_{params['batch_size']}_{params['dropout_rate']}_{random.randint(0, 10000)}.keras"

    # Define a ModelCheckpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')

    # Define the RNN model
    model = Sequential()
    for i, units in enumerate(architecture):
        if i == 0:
            model.add(LSTM(units, input_shape=(params['max_seq_len'], 10*10*4), return_sequences=True))
        else:
            model.add(LSTM(units, return_sequences=(i != len(architecture) - 1)))
        model.add(Dropout(params['dropout_rate']))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dense(train_labels_seq.shape[1] * train_labels_seq.shape[2], activation='softmax'))
    model.add(Reshape((params['max_seq_len'], train_labels_seq.shape[2])))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model with checkpoint saving and TensorBoard logging
    model.fit(train_seq, train_labels_seq, epochs=params['epochs'], batch_size=params['batch_size'], 
              callbacks=[checkpoint_callback, tensorboard_callback], validation_split=0.2)

    # Load the best saved model
    model.load_weights(checkpoint_path)

    # Evaluate the best model
    test_loss, test_acc = model.evaluate(test_seq, test_labels_seq)
    return test_acc, checkpoint_path

max_seq_len = 64
train_seq, train_labels_seq, test_seq, test_labels_seq = whole_sequences(0.95, max_seq_len)

# Define the different architectures and parameters to try
architectures = [
    [256, 128, 64],
    [512, 256, 128],
    [128, 64],
    [256, 128],
    [512, 256, 128, 64]
]
params_list = [
    {'dropout_rate': 0.5, 'max_seq_len': 64, 'epochs': 600, 'batch_size': 32},
    {'dropout_rate': 0.4, 'max_seq_len': 64, 'epochs': 500, 'batch_size': 64},
    {'dropout_rate': 0.3, 'max_seq_len': 64, 'epochs': 400, 'batch_size': 128},
    {'dropout_rate': 0.2, 'max_seq_len': 64, 'epochs': 700, 'batch_size': 256},
    {'dropout_rate': 0.1, 'max_seq_len': 64, 'epochs': 600, 'batch_size': 512}
]

best_acc = 0
best_model_path = None

# Iterate over each architecture and parameter set
for architecture in architectures:
    for params in params_list:
        test_acc, model_path = train_and_evaluate_model(architecture, params, train_seq, train_labels_seq, test_seq, test_labels_seq)
        print(f"Test accuracy for architecture {architecture} with params {params}: {test_acc}")
        with open('models/results.txt', 'a') as f:
            f.write(f"Test accuracy for architecture {architecture} with params {params}: {test_acc}\n")
            f.write(f"Model path: {model_path}\n")
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_path = model_path

print(f"Best model path: {best_model_path}")
print(f"Best test accuracy: {best_acc}")