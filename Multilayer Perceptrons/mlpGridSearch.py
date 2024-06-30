"""
This is a program that will grid search multiple parameters to hopefully create the best neural network for the Pimas
Indians Diabetes dataset to classify whether someone has early onset diabetes or not.
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
# Suppress all warnings as a lot show up that are annoying and do not break the program
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)


# Function to create model, required for Keras Classifier
def create_model(optimizer='adam', activation='relu', neurons=12, init_mode='uniform', sgd=False):
    # Create model
    model = Sequential()
    model.add(Dense(neurons, input_shape=(8,), kernel_initializer=init_mode, activation=activation))
    model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
    # Only compile model if SGD is False, else do not compile model
    if not sgd:
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# Define method to grid search the model, output performance and return the best value for each parameter tuned
def eval_model(model, param_grid):
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(x, y)
    # Summarize results
    print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('%f (%f) with: %r' % (mean, stdev, param))
    results = []
    # Add loop to get the values from the best_params dictionary
    for key in grid_result.best_params_:
        results.append(grid_result.best_params_[key])
    return results


# Fix random seed for reproducibility
seed = 7
tf.random.set_seed(seed)

# Load in the dataset
dataset = np.loadtxt('pimas-indians-diabetes.csv', delimiter=',')
# Split into input [x] and output [y] vars
x = dataset[:, 0:8]
y = dataset[:, 8]
# Create model
model = KerasClassifier(model=create_model, verbose=0)
# Define batch size and epoch for grid search
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
# Summarize tuning of parameters and return best batch_size & epochs
results = eval_model(model, param_grid)
best_batch_size = results[0]
best_epochs = results[1]

# Now tune the training optimization algorithm
model = KerasClassifier(model=create_model, epochs=best_epochs, batch_size=best_batch_size, verbose=0)
# Define optimizer grid search parameters
optimizer = ['SGD', 'RMSprop', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(model__optimizer=optimizer)
# Summarize tuning of learning algorithms
best_opt = eval_model(model, param_grid)

# Now tuning learning rate and momentum which are parameters specific for SGD optimizer models
model = KerasClassifier(model=create_model(sgd=True), loss='binary_crossentropy', optimizer='SGD', epochs=best_epochs,
                        batch_size=best_batch_size, verbose=0)
# Define learning rate and momentum values to grid search
learn_rate = [0.001, 0.01, 0.1, 0.3]
momentum = [0.0, 0.2, 0.4, 0.8, 0.9]
param_grid = dict(optimizer__learning_rate=learn_rate, optimizer__momentum=momentum)
# Output results of the grid search for best learning rate and momentum values
best_rates = eval_model(model, param_grid)

# Next tune network weight initialization hyperparameter
model = KerasClassifier(model=create_model, epochs=best_epochs, batch_size=best_batch_size, verbose=0)
# Define values to grid search
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'he_normal', 'glorot_normal']
param_grid = dict(model__init_mode=init_mode)
# Output results for best weight initialization method
best_init = eval_model(model, param_grid)

# Next tune the neuron activation method
model = KerasClassifier(model=create_model, epochs=best_epochs, batch_size=best_batch_size, verbose=0)
# Define activation functions to be grid searched
activation = ['softmax', 'relu', 'tanh', 'sigmoid', 'linear']
param_grid = dict(model__activation=activation)
# Output results for best activation method
best_act = eval_model(model, param_grid)

# Finally tune number of neurons in the hidden layer
model = KerasClassifier(model=create_model, epochs=best_epochs, batch_size=best_batch_size, verbose=0)
# Define number of neurons to grid search
neurons = [1, 5, 10, 15, 20, 25, 30]
param_grid = dict(model__neurons=neurons)
# Output results for best number of neurons
best_neuron = eval_model(model, param_grid)

# Output the best of each hyperparameter tuning
print('Best epochs: %d\n Best batch size: %d\n Best optimizer: %s\n Best learning rate: %.3f\n Best momentum: %.3f\n'
      'Best init_mode: %s\n Best activation: %s\n Best # of neurons: %d' % (int(best_epochs), int(best_batch_size),
                                                                        best_opt[0], float(best_rates[0]),
                                                                        float(best_rates[1]), best_init[0],best_act,
                                                                        int(best_neuron[0])))


"""
Output since this program takes ~ 15-20 min to run: 
    Best epochs: 100
    Best batch size: 10
    Best optimizer: Nadam
    Best learning rate: 0.001
    Best momentum: 0.800
    Best init_mode: normal
    Best activation: ['relu']
    Best # of neurons: 25
    
* Note grid search does not always have 100% reproducability, even with setting the seed in the program. 
"""