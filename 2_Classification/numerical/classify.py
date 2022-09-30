# Imports
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Input data
# Directory with test images
TEST_FILENAME = 'test.csv'
# Trained model name
MODEL_FILENAME = 'model.h5'

# Load trained model of neural network
model = load_model(MODEL_FILENAME)

# Load test data from CSV file
X = np.loadtxt(TEST_FILENAME, delimiter=",")

# Scale features
min_max_scaler_X = pickle.load(open(MODEL_FILENAME[:-3]+".scx", 'rb'))
X = min_max_scaler_X.transform(X)

# Load model and print it's summary
model = load_model("model.h5")
model.summary()

# Calculate predictions
Y = model.predict(X)

# Scale back features
X = min_max_scaler_X.inverse_transform(X)

# Print predictions
print("\n Classification results:")
for i,x in enumerate(X):
    print("Time={:5.2f}, Au={:4.2f}, Ag={:4.2f}, N={:6.2f}%, Y={:6.2f}%".format(
        x[0], x[1], x[2], Y[i][0]*100, Y[i][1]*100))
