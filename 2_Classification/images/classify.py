# Imports
import tensorflow as tf
import numpy as np
import cv2
import pickle
from imutils import paths

# Input parameters
# Size/resize parameters (they should be the same as used in training)
IMG_SIZE = 28
# Image depth: 1 (grayscale) or 3 (rgb palette)
IMG_DEPTH = 1
# Trained model
MODEL_FILENAME = "model.h5"
# Directory with test images
TEST_IMAGE_DIR = "MNIST-test_images"
# Display (or not) test images with prediction
DISPLAY_IMAGES = True

# Read labels for classes
print("Loading labels...")
with open(MODEL_FILENAME[:-3]+".lbl", 'rb') as f:
    CLASS_LABELS = pickle.load(f)

# Load the trained model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_FILENAME)

# Loop over images and classify
print("Classifying...")
for image_path in sorted(paths.list_images((TEST_IMAGE_DIR))):

    # Load the image and pre-process it
    if IMG_DEPTH == 1:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path)
    image_data = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image_data = image_data.astype("float") / 255.0
    image_data = tf.keras.preprocessing.image.img_to_array(image_data)
    image_data = np.expand_dims(image_data, axis=0)

    # Classify the input image
    prediction = model.predict(image_data)

    # Find and print the winner class and the probability
    winner_class = np.argmax(prediction)
    winner_probability = np.max(prediction)*100

    print("File: {}, prediction - {}: {:.2f}%".format(image_path,
        CLASS_LABELS[winner_class], winner_probability))

    if DISPLAY_IMAGES:
        # Build the label
        label = "{}: {:.2f}%".format(CLASS_LABELS[winner_class], winner_probability)

        # Draw the label on the image
        output_image = cv2.resize(image, (600,600))
        cv2.putText(output_image, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the output image
        cv2.imshow("Prediction", output_image)
        if cv2.waitKey(0) & 0xFF == ord('q'): # Break on 'q' pressed
            break
