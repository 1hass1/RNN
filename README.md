# RNN
# Using a stacked LSTM RNN

# Recurrent Neural Network

# this LSTM will capture the downward and upward trend of google stock price
# LSTM is the most powerful model to do this
# it is not a simple LSTM, but a super robust with 
# some high dimensionality, several layers (a stacked LSTM)
# we will add some dropout regularization to avoid overfitting
# with the most powerful optimizer in the keras library


# we will train the LSTM model on 5 years of the google stock price (2012 to 2016)
# then we will predict January 2017
# we wont predict the exact price, but the trend (upward or downward)



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np   # allows us to make arrays (the only allowed inputs for NNs (data frames cant be inputs for NNs))
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set

# 1st we import the training set as Data Frame using pandas
# then we select the column we need and convert it to an array using numpy 
# (only arrays can be inputs to NNs in keras)
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values
# because we want to input a numpy array and not a simple vector,
# we can do [:, 1], but we must do [:, 1:2]
# this way it is read as a numpy array of 1 column


# Feature Scaling  (2 methods: Standardisation and Normalization)

# in RNNs, it is better to use Normalization
# especially when there is a sigmoid activation function in the output of the RNN
from sklearn.preprocessing import MinMaxScaler   # a class for Normalization
sc = MinMaxScaler(feature_range = (0, 1))        # instance of the Normalization class
training_set_scaled = sc.fit_transform(training_set)
# fit means (here): to apply the min and max of the stock prices to the Normalization formula


# Creating a data structure with 60 timesteps and 1 output
# 60 timesteps is 3 months (20 business days in 1 month) at each time t
# 1 time step is stupid... will lead to overfitting... wont learn
# 20 time steps is not enough to capture some trends
X_train = []   # input: containing the 50 previous stock prices before that financial day
y_train = []   # output containing the stock price the next financial day (business day)
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

# now turn them from lists to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping  (adding a new dimension: The Unit) 
# the  Unit: # of predictors we can use to predict the google stock price at time t+1)
# these predictors are indicators
# we have now 1 indicator: the Open google stock price (can add more)
# we take the 60 previous stock prices to predict the next 1
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# .shape method returns the dimension of the array ... (1198, 60, 1)
# to add a dimension in the numpy array, we need to use the reshape function
# the reshaped input is needed for the RNN
# before, X_train was 2D (1198, 60)
# after reshaping, it will be 3D (3rd dimension corresponding to the indicators)
# example of a new dimension (indicator): amazon stock price,
# bcs it is correlated to the google stock price



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()    # sequence of layers
# called it regressor bcs this time we are predicting a continuous output (price at t+1)
# (essentially, we are doing regression)
# (and classification is to predict a category of a class)


# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))   # Dropout Regularization (0.2 is the dropout rate)
# units = # for LSTM units (cells)
# return_sequences = true, if we want to add another LSTM layer after
# input_shape = includes only the last 2 dimensions (time steps and indicators)of X_train (input)
# (the 1st dimension (the observations) will be automatically taken into account)
# 20% of the neurons will be dropped during each iteration of the training



# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
# this contains the training and test set (combine the data frames version)
# axis = 0, for concatenating them on the vertical axis (axis = 1 for horizontal axis)
# we use concatenation for the original data sets ['Open']
# only inputs must be scaled, bcs the RNN was trained on the scaled values of the training set
# outputs should remain the same

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()












