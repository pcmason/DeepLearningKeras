"""
Program that creates a simple and copmlex CNN and outputs each model's classification accuracy on the CIFAR-10 object
recognition dataset.
"""


from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical


# Method to load data
def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Normalize inputs from 0-255 to 0-1
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # One hot encode outputs
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    num_classes = y_test.shape[1]
    return x_train, y_train, x_test, y_test, num_classes


# Method to create simple CNN
def simple_cnn(num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu', kernel_constraint=MaxNorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=MaxNorm(3)))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=MaxNorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


# Method to create more complex CNN
def complex_cnn(num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', kernel_constraint=MaxNorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=MaxNorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model


# Method to compile model
def compile_model(model, epochs=25, lrate=0.01):
    decay = lrate / epochs
    sgd = SGD(learning_rate=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model


# Method to fit and evaluate model
def eval_model(model, x_train, y_train, x_test, y_test, batch_size, epochs=25):
    # Fit
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)
    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    return scores[1]*100


# Now put all the methods together in the main() function
def main():
    # Create list to hold model accuracies
    accs = list()
    # Load in the data
    x_train, y_train, x_test, y_test, num_classes = load_data()
    # Create the simple CNN model
    simple_model = simple_cnn(num_classes)
    # Compile the CNN model
    simple_model = compile_model(simple_model)
    # Fit and evaluate simple model
    simp_score = eval_model(simple_model, x_train, y_train, x_test, y_test, batch_size=32)
    accs.append(simp_score)
    # Coreate andd compile more complex CNN model
    comp_model = complex_cnn(num_classes)
    comp_model = compile_model(comp_model)
    # Fit and evaluate complex model
    comp_score = eval_model(comp_model, x_train, y_train, x_test, y_test, batch_size=64)
    accs.append(comp_score)
    # Now output the performance of the simple and complex model
    for acc in accs:
        print('Accuracy: %.2f%%' % acc)


# Now run the main method
if __name__ == '__main__':
    main()