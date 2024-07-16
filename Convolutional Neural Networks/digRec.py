"""
This is a program that creates a baseline MLP model, simple, and a more complex Convolutional Neural Network to work
with the MNIST dataset found in Keras. The MNIST problem is a 10-class classification problem that has the model
determine what digit between 0-9 is written in the image.
"""
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical


# Method to load data
def load_data(baseline):
    # Load in MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if baseline:
        # Flatten 28X28 images to a 784 vector for each image
        num_pixels = x_train.shape[1] * x_train.shape[2]
        x_train = x_train.reshape((x_train.shape[0], num_pixels)).astype('float32')
        x_test = x_test.reshape((x_test.shape[0], num_pixels)).astype('float32')
    else:
        num_pixels = 0
        # Reshape to be [samples][width][height][channels]
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32')
        x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32')
    # Normalize inputs from 0-255 to 0-1
    x_train = x_train / 255
    x_test = x_test / 255
    # One hot encode outputs
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    num_classes = y_test.shape[1]
    return x_train, y_train, x_test, y_test, num_pixels, num_classes


# Define baseline model
def baseline_model(num_pixels, num_classes):
    # Create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Create simple CNN model
def simple_cnn(num_classes):
    # Create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Define larger CNN model
def large_cnn(num_classes):
    # Create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Create method to create and evaluate model
def create_eval_model(method, x_train, y_train, x_test ,y_test, num_pixels, num_classes, baseline):
    # Build the model
    if baseline:
        model = method(num_pixels, num_classes)
    else:
        model = method(num_classes)
    # Fit the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=2)
    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    error_rate = 100 - scores[1] * 100
    print('Error: %.2f%%' % (error_rate))
    return error_rate


# Create main method to call
def main():
    # Create lists to keep track of score and model names
    scores, names = list(), list()
    # Load in MNIST data for baseline method
    x_train, y_train, x_test, y_test, num_pixels, num_classes = load_data(baseline=True)

    # Call create_eval_model on baseline_model() and append score and name to lists
    acc = create_eval_model(baseline_model, x_train, y_train, x_test, y_test, num_pixels, num_classes, baseline=True)
    scores.append(acc)
    names.append('Basline model')

    # Load in MNIST data for CNN methods
    x_train, y_train, x_test, y_test, num_pixels, num_classes = load_data(baseline=False)

    # Call create_eval_model on simple_cnn() and append score and name to lists
    acc = create_eval_model(simple_cnn, x_train, y_train, x_test, y_test, num_pixels, num_classes, baseline=False)
    scores.append(acc)
    names.append('Simple CNN')

    # Call create_eval_model on the large_cnn() and append score and name to lists
    acc = create_eval_model(large_cnn, x_train, y_train, x_test, y_test, num_pixels, num_classes, baseline=False)
    scores.append(acc)
    names.append('Large CNN')

    for i in range(len(scores)):
        print('%s error: %.2f%%' % (names[i], scores[i]))


if __name__ == '__main__':
    main()
