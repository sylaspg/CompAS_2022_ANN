# Imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
import cv2
import pickle
import random
import os
import nn  # Module with network definitions
from timeit import default_timer as timer # For execution time calculation


# Input parameters
# Random seed
RANDOM_SEED = 100
# Batch size
BS = 32
# Images will be resized to (SIZE x SIZE) dimension
IMG_SIZE = 28
# Image depth: 1 (grayscale) or 3 (rgb palette)
IMG_DEPTH = 1
# Fraction of validation data
VALID_SPLIT = 0.2
# No of epochs for training
EPOCHS = 50
# Directory with training images
# Keep training images  for each class in the subdirectories:
# TRAIN_IMAGE_DIR/class-0-class0name
# TRAIN_IMAGE_DIR/class-1-class1name, etc.
TRAIN_IMAGE_DIR = "MNIST-train_images"
# Filename for trained model
MODEL_FILENAME = "model.h5"
# Use (or not) GPU (Nvidia with CUDA only)
USE_GPU = True


# Initializations
start_time = timer()
if not USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Random seed, for reproductible results (works stable for CPU only)
tf.keras.utils.set_random_seed(RANDOM_SEED) # TF >= 2.7.0

# Initialize lists for data and labels
print("Loading images...")
X = []
Y = []
labels = []

# Grab the image paths and randomly shuffle them
image_paths = list(paths.list_images(TRAIN_IMAGE_DIR))
random.shuffle(image_paths)

# Loop over the input images
for image_path in image_paths:
    # Load the image and pre-process it
    if IMG_DEPTH == 1:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.preprocessing.image.img_to_array(image)

    # Extract the class label (numeric and text) from the image path
    dirname = image_path.split(os.path.sep)[-2]
    dirname_list = dirname.split("-")

    if dirname_list[0] != "class":
        # File not in "class-*" directory, skip it
        continue

    # Y (class index) is the number in the directory name, after first "-"
    class_idx = int(dirname_list[1])
    # Text label is the text in the directory name, after second "-"
    try:
        label = dirname_list[2]
    except KeyError:
        label = int(dirname_list[1])

    # Store image and labels in lists
    X.append(image)
    Y.append(class_idx)
    labels.append(label)
    
print("  {} images loaded.".format(len(X)))

# Get the unique classes names
classes = np.unique(labels)

# Save the text labels to disk as pickle
with open(MODEL_FILENAME[:-3]+".lbl", "wb") as f:
    f.write(pickle.dumps(classes))

# Determine number of classes
no_classes = len(classes)

# Convert the labels from integers to category vectors
Y = tf.keras.utils.to_categorical(Y, num_classes=no_classes)

# Scale the raw pixel intensities to the [0, 1] range
X = np.array(X, dtype="float") / 255.0

# Initialize the model
print("Compiling model...")
model = nn.SmallerVGGNet.build(width=IMG_SIZE, height=IMG_SIZE, depth=IMG_DEPTH, classes=no_classes)

# LeNet is not a good choice for general image classification, since it works on 32x32 pixel images,
# use for testing only
#model = nn.LeNet5.build(width=IMG_SIZE, height=IMG_SIZE, depth=IMG_DEPTH, classes=no_classes)

# Fully Connected - also not a very good choice, use for testing only; tune the hidden_units
#model = nn.FullyConnectedForImageClassisfication.build(width=IMG_SIZE, height=IMG_SIZE, depth=IMG_DEPTH,
#                                                       hidden_units=1000, classes=no_classes)
model.summary()

# Select the loss function
if no_classes == 2:
    loss = "binary_crossentropy"
else:
    loss = "categorical_crossentropy"

# Compile model
model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])

# Train the network
print("Training network...")
H = model.fit(X, Y, batch_size=BS, epochs=EPOCHS, validation_split=VALID_SPLIT)

# Evaluate the model
print("\nEvaluate model:")
model.evaluate(X, Y)

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

# Save model to disk
print("Saving model and plots...")
model.save(MODEL_FILENAME)

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(range(EPOCHS), H.history["loss"], label="train_loss")
plt.plot(range(EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(range(EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(range(EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(MODEL_FILENAME[:-3]+".png")

# Finishing
end_time = timer() - start_time
print("... finished in {:.2f} seconds.".format(end_time))
