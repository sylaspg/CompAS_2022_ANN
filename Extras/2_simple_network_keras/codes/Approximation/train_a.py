# Version with automatic train/validation split

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Alternate imports - only required functions from TF:
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense


# CSV file name with train data
DATAFILE = "train.csv"

# No of epochs for training
EPOCHS = 10000

# Validation / train split
SPLIT = 0.2

# Size of input vector 
#(number of neurons in the input layer)
Nin = 1

# Load train data from CSV file
data = np.loadtxt(DATAFILE, delimiter=",")

# Size of output vector 
# (number of neurons in the output layer)
Nout = data.shape[1] - Nin

# Divide into X and Y vectors
X = data[:, :Nin]
Y = data[:, Nin:]

# Create model
# https://keras.io/layers/core/
# https://keras.io/activations/
# One-directional network
model = tf.keras.models.Sequential()
# Input layer of dimension Nin
model.add(tf.keras.layers.InputLayer
          (input_shape=(Nin, )))
# Hidden layer: 3 neurons
model.add(tf.keras.layers.Dense(3, activation='tanh'))
# Alternative for two above layers:
#model.add(tf.keras.layers.Dense(3, input_dim=Nin, activation='tanh'))
# Hidden layer: 5 neurons
model.add(tf.keras.layers.Dense(5, activation='tanh'))
# Output layer: Nout neurons
model.add(tf.keras.layers.Dense(Nout, activation='sigmoid'))

# Compile model
# Loss functions: https://keras.io/losses/
# Optimizers: https://keras.io/optimizers/
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model
H = model.fit(X, Y, epochs=EPOCHS, validation_split=SPLIT)

# Evaluate the model
print("Evaluate model:")
model.evaluate(X, Y)

# Alternative:
#test_score = model.evaluate(X, Y, verbose=0)
#print("Loss on complete training set = {:.15f}".format(test_score))

# Save model to file
model.save("model.h5")

# Plot the train and validation losses
plt.style.use("ggplot")
plt.figure()
plt.plot(range(EPOCHS), H.history["loss"], 
        label="Train loss")
plt.plot(range(EPOCHS), H.history["val_loss"], 
        label="Validation loss")
plt.title("Train / validation loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("model.png")
