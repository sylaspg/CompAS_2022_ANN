# Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle

# Input data
# CSV file name with train data
DATA_FILENAME = "data.csv"
# Filename for output (trained) model (in HDF5 format)
MODEL_FILENAME = "model.h5"
# Size of input vector (number of neurons in the input layer)
Nin = 3
# No of epochs for training
EPOCHS = 1000
# Validation / train split
SPLIT = 0.1
# Seed for random number generator - for reproductible results
RANDOM_SEED = 100

# Initializations
# Set random seed
tf.random.set_seed(RANDOM_SEED)

# Load data - now separately for X and Y, since Y are string labels
X = np.loadtxt(DATA_FILENAME, delimiter=",", usecols=tuple(range(Nin)))
Y = np.loadtxt(DATA_FILENAME, delimiter=",", usecols=(Nin), dtype=str)

# Strip spaces around Y data
Y = [y.strip() for y in Y]

# Number of classes for prediction (number of neurons in the output layer)
no_classes = len(np.unique(Y))

# Normalize X and save scaling object for future use
min_max_scaler_X = preprocessing.MinMaxScaler().fit(X)
X = min_max_scaler_X.transform(X)
pickle.dump(min_max_scaler_X, open(MODEL_FILENAME[:-3]+".scx", 'wb'))

# Transform labels to integers
label_encoder = preprocessing.LabelEncoder()
Y = label_encoder.fit_transform(Y)

# Transform labels to 0-1 encoding
# N -> [1, 0]
# Y -> [0, 1]
Y = tf.keras.utils.to_categorical(Y, no_classes)
# or skip encoding, but use "sparse_categorical_crossentropy" instead of
# categorical_crossentropy or binary_crossentropy as loss later in compile() 

# Create model
# https://keras.io/layers/core/
# https://keras.io/activations/
# One-directional network
model = tf.keras.models.Sequential()
# Input layer of dimension Nin
model.add(tf.keras.layers.InputLayer(input_shape=(Nin, )))
# Hidden layer: 3 neurons
model.add(tf.keras.layers.Dense(4, activation='tanh'))
# Alternative for two above layers:
#model.add(tf.keras.layers.Dense(2, input_dim=Nin, activation='tanh'))
# Hidden layer: 5 neurons
#model.add(tf.keras.layers.Dense(4, activation='tanh'))
# Output layer: Nout neurons
model.add(tf.keras.layers.Dense(no_classes, activation='softmax'))

# Compile model
# Loss functions: https://keras.io/losses/
# Optimizers: https://keras.io/optimizers/
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

# Fit the model
H = model.fit(X, Y, epochs=EPOCHS, validation_split=SPLIT)

# Evaluate the model
print("\nEvaluate model:")
model.evaluate(X, Y)

# Alternative:
#test_score = model.evaluate(X, Y, verbose=0)
#print("Loss (MSE) on complete training set = {:.15f}".format(test_score))

# Classification scores (recall, precision and f1)
print("\nScores:")
# We need predictions to calculate metrics
Ypred = model.predict(X)

metric = tf.keras.metrics.Recall()
metric.update_state(Y[:,0], Ypred[:,0])
recall = metric.result()
print("Recall    = {:.3f}".format(recall))

metric = tf.keras.metrics.Precision()
metric.update_state(Y[:,0], Ypred[:,0])
precision = metric.result()
print("Precision = {:.3f}".format(precision))

f1 = 2*(precision*recall)/(precision+recall)
print("F1        = {:.3f}".format(f1))

# Save model to file
model.save(MODEL_FILENAME)

# Plot the train and validation losses/accuracies
plt.style.use("ggplot")
plt.figure()
plt.plot(range(EPOCHS), H.history["loss"], label="Train loss")
plt.plot(range(EPOCHS), H.history["accuracy"], label="Train accuracy")
plt.plot(range(EPOCHS), H.history["val_loss"], label="Validation loss")
plt.plot(range(EPOCHS), H.history["val_accuracy"], label="Validation accuracy")
plt.title("Train / validation loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("model.png")
