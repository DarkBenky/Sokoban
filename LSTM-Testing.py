import tensorboard as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape , Dropout
import numpy as np
import random
import json

# check for gpu availability
# print("Num GPUs Available: ", len(tf.test.is_gpu_available()))

tensorboard_callback = TensorBoard(log_dir='./logs-LSTM', histogram_freq=1)

from data_processing import remove_duplicate_from_array , load_seq

seq_length = 8

data , labels = remove_duplicate_from_array(load_seq(seq_length))

data = np.array(data)
labels = np.array(labels)

split = 0.8

name = 'LSTM-' + str(random.randint(0, 1000)) + ".keras"

checkpoint = ModelCheckpoint('models/LSTM/V2'+name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model = Sequential([
    LSTM(128, input_shape=(seq_length, 10*10*4) , return_sequences=True),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(data, labels, epochs=1000, batch_size=64, validation_split=0.2
            , callbacks=[checkpoint, tensorboard_callback])

# load the best model
model.load_weights('models/LSTM/'+name)

# Evaluate the best model
test_loss, test_acc = model.evaluate(data, labels)

print(f"Test accuracy: {test_acc}")
print(f"Test loss: {test_loss}")
print(f"Model saved as {name}")

with open('models/LSTM/V2/model.json', 'w') as f:
    data = {
        'name': name,
        'accuracy': test_acc,
        'loss': test_loss
    }
    json.dump(data, f)

