# Disable TensorFlow annoying messages:
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# ... and then import tensorflow 

# Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle
import tensorflow_addons as tfa

# Input data
# CSV file name with train data
DATA_FILENAME = "data.csv"
# Filename for output (trained) model (HDF5 format)
MODEL_FILENAME = "model.h5"
# No of epochs for training
EPOCHS = 1000
# Validation / train split
SPLIT = 0.2
# Size of input vector (number of neurons in the input layer)
Nin = 3
# Size of output vector (number of neurons in the output layer)
Nout = 1
# Seed for random number generator - for reproductible results
RANDOM_SEED = 100

#Initializations
# Set random seed (TF >= 2.7, CPU only)
tf.random.set_seed(100)

# Load data
data = np.loadtxt(DATA_FILENAME, delimiter=",")

# Divide into X and Y vectors
X = data[:, :Nin]
Y = data[:, Nin:]

# Normalize data and save scaling objects for future use
min_max_scaler_X = preprocessing.MinMaxScaler().fit(X)
X = min_max_scaler_X.transform(X)
pickle.dump(min_max_scaler_X, open(MODEL_FILENAME[:-3]+".scx", 'wb'))

min_max_scaler_Y = preprocessing.MinMaxScaler().fit(Y)
Y = min_max_scaler_Y.transform(Y)
pickle.dump(min_max_scaler_Y, open(MODEL_FILENAME[:-3]+".scy", 'wb'))

# Create model
# https://keras.io/layers/core/
# https://keras.io/activations/
# One-directional network
model = tf.keras.models.Sequential()
# Input layer of dimension Nin
model.add(tf.keras.layers.InputLayer(input_shape=(Nin, )))
# Hidden layer: 3 neurons
model.add(tf.keras.layers.Dense(2, activation='tanh'))
# Alternative for two above layers:
#model.add(tf.keras.layers.Dense(2, input_dim=Nin, activation='tanh'))
# Hidden layer: 5 neurons
model.add(tf.keras.layers.Dense(4, activation='tanh'))
# Output layer: Nout neurons
model.add(tf.keras.layers.Dense(Nout, activation='sigmoid'))

# Compile model
# Loss functions: https://keras.io/losses/
# Optimizers: https://keras.io/optimizers/
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model
H = model.fit(X, Y, epochs=EPOCHS, validation_split=SPLIT)

# Evaluate the model
print("\nEvaluate model:")
model.evaluate(X, Y)

# Alternative:
#test_score = model.evaluate(X, Y, verbose=0)
#print("Loss (MSE) on complete training set = {:.15f}".format(test_score))

# Regression scores (R squared and RMSE)
print("\nScores:")
# We need predictions to calculate metrics
Ypred = model.predict(X)

metric = tfa.metrics.RSquare()
metric.update_state(Y[:,0], Ypred[:,0])
r2 = metric.result()
print("R^2  = {:.3f}".format(r2))

metric = tf.keras.metrics.RootMeanSquaredError()
metric.update_state(Y[:,0], Ypred[:,0])
rmse = metric.result()
print("RMSE = {:.3f}".format(rmse))

# Save model to file
model.save(MODEL_FILENAME)

# Plot the train and validation losses
plt.style.use("ggplot")
plt.figure()
plt.plot(range(EPOCHS), H.history["loss"], label="Train loss")
plt.plot(range(EPOCHS), H.history["val_loss"], label="Validation loss")
plt.title("Train / validation loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("model.png")
