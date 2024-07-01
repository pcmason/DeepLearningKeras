"""
Simple program that trains a MLP on a classification dataset and outputs the history of the accuracy & loss metrics
for test and validation sets over training epochs.
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np


# Create method to plot history graphs
def summarize_plot(history, hist1, hist2, title, yLabel):
    plt.plot(history.history[hist1])
    plt.plot(history.history[hist2])
    plt.title(title)
    plt.ylabel(yLabel)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# Load pima dataset
dataset = np.loadtxt('pimas-indians-diabetes.csv', delimiter=',')
# Split into x and y vars
x = dataset[:, 0:8]
y = dataset[:, 8]
# Create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(x, y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
# List all data in history
print(history.history.keys())
# Summarize history for accuracy
summarize_plot(history, 'accuracy', 'val_accuracy', 'Model Accuracy', 'accuracy')
# Summarize history for loss
summarize_plot(history, 'loss', 'val_loss', 'Model Loss', 'loss')
