"""
Program that creates a LSTM with variable length input sequences to one character output so that the LSTM model
can learn the alphabet and predict the next letter based on a sequence of previous letters.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Fix random seed for reproducability
np.random.seed(7)
tf.random.set_seed(7)
# Define the alphabet (dataset)
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Create mapping of characters to integers (0-25) and the reverse
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))


# Define method that prepared dataset of input to output pairs encoded as integers
def prep_data():
    num_inputs = 1000
    # Max length of each sequence
    max_len = 5
    dataX, dataY = [], []
    for i in range(num_inputs):
        # get start and end of the sequence
        start = np.random.randint(len(alphabet)-2)
        end = np.random.randint(start, min(start + max_len, len(alphabet)-1))
        # now create the sequence
        sequence_in = alphabet[start: end+1]
        sequence_out = alphabet[end + 1]
        dataX.append([char_to_int[char] for char in sequence_in])
        dataY.append(char_to_int[sequence_out])
        #print(sequence_in, '->', sequence_out)
    return dataX, dataY, max_len


# Create method to reshape x and y variables
def reshape_data(dataX, dataY, max_len):
    # Convert list of lists to array and pad sequences if needed
    X = pad_sequences(dataX, maxlen=max_len, dtype='float32')
    # Reshape X to be [samples, timesteps, features]
    X = np.reshape(X, (X.shape[0], max_len, 1))
    # Normalize
    X = X / float(len(alphabet))
    # One hot encode the output variable
    y = to_categorical(dataY)
    return X, y


# Method to create, fit and evaluate LSTM model
def create_LSTM(X, y):
    batch_size = 1
    model = Sequential()
    model.add(LSTM(32, input_shape=(X.shape[1], 1)))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=500, batch_size=batch_size, verbose=2)
    # Summarize performance of the model
    scores = model.evaluate(X, y, verbose=0)
    print("Model Accuracy: %.2f%%" % (scores[1]*100))
    return model


# Demonstrate some model predictions
def dem_model(model, dataX, max_len):
    for i in range(20):
        # Create subsequence of letters for model
        pattern_index = np.random.randint(len(dataX))
        pattern = dataX[pattern_index]
        # Pad, reshape and normalize x
        x = pad_sequences([pattern], maxlen=max_len, dtype='float32')
        x = np.reshape(x, (1, max_len, 1))
        x = x / float(len(alphabet))
        # Now that x is normalized and ready get predictions
        prediction = model.predict(x, verbose=0)
        # Get int value of next predicted letter
        index = np.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        # Clean way of outputting predictions
        print(seq_in, '->', result)


# Now tie all the code together with a main() method
def main():
    # First get the x, y and max sequence length
    dataX, dataY, max_len = prep_data()
    # Now reshape the data to get X and y
    X, y = reshape_data(dataX, dataY, max_len)
    # Create and evaluate the LSTM model to learn the alphabet
    model = create_LSTM(X, y)
    # Finally demonstrate the effectiveness of the model
    dem_model(model, dataX, max_len)


if __name__ == '__main__':
    main()