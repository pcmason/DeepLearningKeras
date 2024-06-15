"""Program that runs the saved model.keras from the keras_first-network.py program"""

import numpy as np
from tensorflow.keras.models import load_model

# Load in model
model = load_model('model.keras')
# Summarize model
model.summary()
# Load data
dataset = np.loadtxt('pimas-indians-diabetes.csv', delimiter=',')
# Split into input and output variables
x = dataset[:, 0:8]
y = dataset[:, 8]
# Evaluate loaded model
score = model.evaluate(x, y, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], score[1] * 100))