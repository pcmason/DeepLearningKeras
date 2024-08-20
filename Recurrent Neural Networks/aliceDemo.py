"""
This program takes the best model from textGenAlice [should be saved in this directory] and uses it to generate text
based on Lewis Carrol's Alice in Wonderland novel. While the text generated is not necessarily coherent, it is still
surprising how successful such a naive model can be.
"""
import sys
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from textGenAlice import load_txt, prep_data


# Create method to create large LSTM model and load it in with best weights
def load_lstm_model(filename, x, y):
    model = Sequential()
    model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    # Load the network weights
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


# Now pick a random seed
def pick_seed(dataX, int_to_char):
    start = np.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    print("Seed:")
    print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    return pattern


# Create method to generate text
def gen_text(model, pattern, int_to_char, n_vocab):
    for i in range(1000):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        # Re adjust the window of the subsequence
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("\nDone.")


# Create main method to tie all together
def main():
    # Load in text
    text = load_txt('wonderland.txt')
    # Get x, y, dataX and chars
    x, y, dataX, chars, n_vocab = prep_data(text)
    # Load in the model with already learned weights
    model = load_lstm_model('weights-improvement-16-1.5530-bigger.keras', x, y)
    # Now use chars to make int_to_char dict
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    # use dataX to get the random seed
    pattern = pick_seed(dataX, int_to_char)
    # Finally put it all together to generate text
    gen_text(model, pattern, int_to_char, n_vocab)


if __name__ == '__main__':
    main()