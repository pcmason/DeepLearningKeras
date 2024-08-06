"""
Program that uses the IMDB movie review dataset from the Keras TF library and creates two simple models. The problem is
that the movie reviews in the dataset are all either good or bad reviews and the models must determine the sentiment of
the review.

Will create a simple MLP model and 1D CNN model to be evaluated on the IMDB dataset. Both should achieve an accuracy of
~87%.
"""
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Conv1D, MaxPooling1D
from tensorflow.keras.preprocessing import sequence

# Set constants for model
TOP_WORDS = 5000
MAX_WORDS = 500


# Method to load in imdb dataset
def load_imdb():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=TOP_WORDS)
    # Truncate or pad all reviews to be 500 words in length
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_WORDS)
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_WORDS)
    return x_train, y_train, x_test, y_test


# Create method for simple MLP model
def simple_mlp():
    model = Sequential()
    model.add(Embedding(TOP_WORDS, 32, input_length=MAX_WORDS))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


# Now create model for 1D CNN:
def one_dim_cnn():
    model = Sequential()
    model.add(Embedding(TOP_WORDS, 32, input_length=MAX_WORDS))
    # These 2 layers are the big difference between the CNN and the MLP
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


# Create method to evaluate models
def eval_model(model, x_train, y_train, x_test, y_test):
    # Epochs is small since model overfits quickly, batch size is large due to large size of training set
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=128, verbose=2)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print('Accuracy: %.2f%%' % (scores[1]*100))
    return scores[1]*100


# Create the main method that runs all functions
def main():
    # Load in dataset
    x_train, y_train, x_test, y_test = load_imdb()
    # Create list to store answers
    answers = list()
    # Create simple MLP model and evaluate it
    mlp = simple_mlp()
    mlp_answer = eval_model(mlp, x_train, y_train, x_test, y_test)
    answers.append(mlp_answer)
    # Create 1D CNN model and evaluate it
    cnn = one_dim_cnn()
    cnn_answer = eval_model(cnn, x_train, y_train, x_test, y_test)
    answers.append(cnn_answer)
    # Now loop through answers and output accuracy of first MLP then CNN model
    for answer in answers:
        print('Accuracy: %.2f%%' % answer)


if __name__ == '__main__':
    main()