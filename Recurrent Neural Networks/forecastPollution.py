"""
Program that uses the Air Quality dataset of the air pollution in Beijing over 5 years. The dataset will be read-in
and modified to suit the model. Then will use the 3 previous hours to forecast what the pollution for the next hour
will be. Model performs with a RMSE of ~27.
"""
from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM


# Convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # Input sequence (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # Forecast sequence (t, t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # Drop rows with NAN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# Create method to read data from .csv file
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')


# Create method to load in raw.csv and save it as pollution.csv
def load_data(csv_file):
    dataset = pd.read_csv(csv_file, parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
    dataset.drop('No', axis=1, inplace=True)
    # Manually specify column names
    dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'
    # Mark all NA values with 0
    dataset['pollution'].fillna(0, inplace=True)
    # Drop first 24 hours
    dataset = dataset[24:]
    # Save to file
    dataset.to_csv('pollution.csv')


def main():
    # Load dataset
    load_data('raw.csv')
    dataset = pd.read_csv('pollution.csv', header=0, index_col=0)
    values = dataset.values
    # Label encode wind direction
    encoder = LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])
    # Ensure all data is float
    values = values.astype('float32')
    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # Specify number of lag hours
    n_hours = 3
    n_features = 8
    # Frame as supervised learning
    reframed = series_to_supervised(scaled, n_hours, 1)
    #print(reframed.shape)

    # Split into train and test sets
    values = reframed.values
    n_train_hours = 365 * 24
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # Split into input and output
    n_obs = n_hours * n_features
    train_x, train_y = train[:, :n_obs], train[:, -n_features]
    test_x, test_y = test[:, :n_obs], test[:, -n_features]
    # Reshape input to be 3D [samples, timesteps, features]
    train_x = train_x.reshape((train_x.shape[0], n_hours, n_features))
    test_x = test_x.reshape((test_x.shape[0], n_hours, n_features))

    # Design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # Fit network
    history = model.fit(train_x, train_y, epochs=50, batch_size=72,
                        validation_data=(test_x, test_y), verbose=2, shuffle=False)
    # Plot history
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Test')
    plt.legend()
    plt.show()

    # Make a prediction
    yhat = model.predict(test_x)
    test_x = test_x.reshape((test_x.shape[0], n_hours*n_features))
    # Invert scaling for forecast
    inv_yhat = concatenate((yhat, test_x[:, -7:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # Invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_x[:, -7:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # Calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)


if __name__ == '__main__':
    main()
