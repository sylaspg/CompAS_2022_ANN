# Imports
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pickle

# Input data
# CSV file name with test data
TEST_FILENAME = "test.csv"
# Trained model filename
MODEL_FILENAME = "model.h5"

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

# Scale back predictions
min_max_scaler_Y = pickle.load(open(MODEL_FILENAME[:-3]+".scy", 'rb'))
Y = min_max_scaler_Y.inverse_transform(Y)

# Scale back features
X = min_max_scaler_X.inverse_transform(X)

#for i in range(len(Y)):
#    print(X[i],Y[i])

# Print predictions
print("\nPrediction results:")
for i,x in enumerate(X):
    print("d={:3.1f}, A={:13.2f}, omega={:8.2f}, MAX={:6.2f}".format(
        x[0], x[1], x[2], Y[i][0]))


# Perform "massive" prediction and plot graphs
# Plots will be for fixed d (from 2 to 8 with step 1),
# while A and omega will be divided into 100 intervals each
print("\nMassive prediction...")

# Ranges for variables
d = np.arange(start=2, stop=8, step=1)
A = np.arange(start=8.2e+08, stop=2.5e+09, step=1.68E+07)
omega = np.arange(start=12500, stop=20000, step=75)

# Collect sets of three variables and transform
X = []
for x0 in d:
    for x1 in A:
        for x2 in omega:
            X.append([x0, x1, x2])
X = np.array(X)
X = min_max_scaler_X.transform(X)

# Predict and transform back
Y = model.predict(X)
Y = min_max_scaler_Y.inverse_transform(Y)

# Empty list for plots for different d's
plots = []

# Collect data for plots
for i0,x0 in enumerate(d):
    # len(A) x len(omega) 2D list
    single_plot = [[None for i in range(len(A))] for j in range(len(omega))]
    for i1,x1 in enumerate(A):
        for i2,x2 in enumerate(omega):
            # Index of Y in a flat Y list
            index = i0 * len(A) * len(omega) + i1 * len(omega) + i2
            single_plot[i1][i2] = Y[index]
    plots.append(single_plot)

# Plot plots for different d's
for i,d_val in enumerate(d):
    ztable = plots[i]

    plt.imshow(ztable, cmap='hot', interpolation='nearest')
    plt.rcParams['axes.grid'] = False
    plt.colorbar().set_label("M")

    # Replace [0,1,2,...100] ticks by the actual values
    plt.xticks([0, len(A)/2, len(A)],[min(A), np.mean(A), max(A)])
    plt.yticks([0, len(omega)/2, len(omega)],[min(omega), np.mean(omega), max(omega)])

    plt.xlabel("A")
    plt.ylabel("omega")
    plt.title("Prediction of the absorption maxima position for d={}".format(d_val))
    plt.savefig("figure"+str(d_val)+".png")
    plt.show()
    plt.clf()

print("Finished.")
