"""
Program that uses Carrol's Alice in Wonderland novel from projectgutenburg.com to train a LSTM that generates text.
This file will be used for training and saving the best performing model, the file aliceDemo.py will be used to
demonstrate the model's capability to generate text.
"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical


# Method to load in ascii text and convert to lowercase
def load_txt(filename):
    raw_text = open(filename, 'r', encoding='utf-8').read()
    raw_text = raw_text.lower()
    return raw_text


# Create method to prepare the data for the model
def prep_data(raw_text):
    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    # Summarize loaded data
    n_chars = len(raw_text)
    n_vocab = len(chars)
    print("Total Characters: ", n_chars)
    print("Total Vocab: ", n_vocab)
    # Prepare dataset of input to output pairs encoded as integers
    seq_length = 100
    dataX, dataY = [], []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)
    # Reshape x to be [samples, time steps, features]
    x = np.reshape(dataX, (n_patterns, seq_length, 1))
    # Normalize
    x = x / float(n_vocab)
    # One hot encode the output variable
    y = to_categorical(dataY)
    return x, y, dataX, chars, n_vocab


# Method to define the multi-layer LSTM model
def lstm_model(x, y):
    model = Sequential()
    model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    # Second layer
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


# Finally create method to define checkpoint and fit model
def checkpoint_fit_model(model, x, y):
    # Define checkpoint
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # Fit the model
    model.fit(x, y, epochs=50, batch_size=64, callbacks=callbacks_list)


# Now tie it all together in main
def main():
    # Load in book text
    text = load_txt('wonderland.txt')
    # Prepare x and y for the model
    x, y, dataX, chars, n_vocab = prep_data(text)
    # Now create the LSTM model
    model = lstm_model(x, y)
    # Finally fit the model and save all the best performing models
    checkpoint_fit_model(model, x, y)


if __name__ == '__main__':
    main()
