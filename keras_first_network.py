"""First neural network with keras to make prediction on the Pimas Indians Diabetes dataset
Extensions implemented to program:
    * Saves the model to be loaded and run on runSavedModel.py.
    * Summarizes model and creates plot of model layers.
    * Separates train & test datasets.
    * Plots the loss and accuracy of the model over the training epochs.
    * Created the model using the Keras functional API.
    * Tuned epoch and batch_size variables.

"""
from numpy import loadtxt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Create a method to plot the history and loss graphs for the trained model
def plot_ml_graph(hist, stat, val_stat, title):
    plt.plot(hist.history[stat])
    plt.plot(hist.history[val_stat])
    plt.title(title)
    plt.ylabel(stat)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# Load Pimas dataset
dataset = loadtxt('pimas-indians-diabetes.csv', delimiter=',')
# Split into input and output variables
x = dataset[:, 0:8]
y = dataset[:, 8]
# Further split into train and test splits
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state= 1, shuffle=True)
# Define keras model, updated to use the keras API
visible = Input(shape=(8,))
hidden1 = Dense(12, activation='relu')(visible)
hidden2 = Dense(8, activation='relu')(hidden1)
output = Dense(1, activation='sigmoid')(hidden2)
model = Model(inputs=visible, outputs=output)
# Compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the keras model on the dataset
history = model.fit(x, y, validation_split=0.25, epochs=300, batch_size=20, verbose=0)
# Now print the history for accuracy and loss
plot_ml_graph(history, 'accuracy', 'val_accuracy', 'model accuracy')
plot_ml_graph(history, 'loss', 'val_loss', 'model loss')
# Evaluate the keras model
_, accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %.3f' % (accuracy*100))
# Save the model
model.save('model.keras')
print('Saved model to disk.')
# Next summarize and visualize the model
print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# Make class predictions with the model
# predictions = (model.predict(x) > 0.5).astype(int)
# Summarize first 5 cases
# for i in range(5):
#    print('%s => %d (expected %d)' % (x[i].tolist(), predictions[i], y[i]))
