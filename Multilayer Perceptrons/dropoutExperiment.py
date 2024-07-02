"""
Program that runs a normal MLP model, one with dropout performed on the visible layer and one where dropout is
performed on the hidden and output layers. The dataset used is the sonar.csv dataset.
"""
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers import SGD
from scikeras.wrappers import  KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Load sonar dataset
dataframe = pd.read_csv('sonar.csv', header=None)
dataset = dataframe.values
# Split into x and y vars
x = dataset[:, 0:60]
y = dataset[:, 60]
# Encode class values [y] as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)


# Create baseline model with no dropout
def create_baseline():
    # Create model
    model = Sequential()
    model.add(Dense(60, input_dim=60, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    sgd = SGD(learning_rate=0.01, momentum=0.8)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


# Create model with dropout at visible layer
def create_vl_dropout():
    # Create model
    model = Sequential()
    # Dropout before input layer
    model.add(Dropout(0.2, input_dim=60))
    model.add(Dense(60, activation='relu', kernel_constraint=MaxNorm(3)))
    model.add(Dense(30, activation='relu', kernel_constraint=MaxNorm(3)))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model with higher learning rate and momentum
    sgd = SGD(learning_rate=0.1, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


# Create model with dropout between hidden layers and output layer
def create_hl_dropout():
    # Create model
    model = Sequential()
    model.add(Dense(60, input_dim=60, activation='relu', kernel_constraint=MaxNorm(3)))
    # Add first dropout between hidden layers
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='relu', kernel_constraint=MaxNorm(3)))
    # Add dropout between hidden layers and output layer
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    sgd = SGD(learning_rate=0.1, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


# Method to evaluate models
def eval_model(model, model_type):
    # Base code to evaluate each model
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(model=model, epochs=300, batch_size=16, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(pipeline, x, encoded_y, cv=kfold)
    result_string = '%s: %.2f%% (%.2f%%)' % (model_type, results.mean() * 100, results.std() * 100)
    print(result_string)
    # Return the string result to output at of program for cleanliness and easy comparison
    return result_string


results = list()
# Evaluate the baseline model
results.append(eval_model(create_baseline, 'Baseline'))
# Evaluate the model with dropout on the input layer
results.append(eval_model(create_vl_dropout, 'Input Layer Dropout'))
# Evaluate model with dropout between hidden layers and before output layer
results.append(eval_model(create_hl_dropout, 'Hidden Layer Dropout'))
# Output results all at end of program
for result in results:
    print(result)

