"""
Program that loads in the saved JSON model.json & w.weights.h5 & the keras model and evaluates them both. Since they are
the same models they should have the same performance.
"""
from tensorflow.keras.models import model_from_json, load_model
import numpy as np

# Fix random seed
np.random.seed(7)
# Load in dataset
dataset = np.loadtxt('pimas-indians-diabetes.csv', delimiter=',')
# Split into x and y vars
x = dataset[:, 0:8]
y = dataset[:, 8]

# Load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# Load weights into loaded model
loaded_model.load_weights('w.weights.h5')
print('Loaded model from disk [JSON].')

# Evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(x, y, verbose=0)
print('%s: %.2f%%' % (loaded_model.metrics_names[1], score[1]*100))

# Load model using keras
model = load_model('model.h5')
# Summarize model
model.summary()
# Evaluate model
score_keras = model.evaluate(x, y, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], score_keras[1]*100))
