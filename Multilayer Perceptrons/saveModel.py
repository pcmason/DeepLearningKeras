"""
Program that saves MLP network to JSON and to keras and loads in the models as well.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
# print(tf.__version__)

# Fix random seed
np.random.seed(7)
# load in the pimas dataset
dataset = np.loadtxt('pimas-indians-diabetes.csv', delimiter=',')
# Split into input and output vars
x = dataset[:, 0:8]
y = dataset[:, 8]
# Create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit model
model.fit(x, y, epochs=150, batch_size=10, verbose=0)
# Evaluate model
scores = model.evaluate(x, y, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))

# Serialize model to JSON
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
# Serialize weights to HDF5
model.save_weights('w.weights.h5')
print('Saved model to disk [JSON].')

# Save model & architecture to single file using keras
model.save('model.h5')
print('Saved model to disk [keras].')

