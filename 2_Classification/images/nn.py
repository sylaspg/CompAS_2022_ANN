import tensorflow as tf

class FullyConnectedForMnist:
    '''
    Simple NN for MNIST database. INPUT => FC/RELU => FC/SOFTMAX
    '''
    def build(hidden_units):
        model = FullyConnectedForImageClassisfication.build(28, 28, 1, hidden_units, 10)
        return model


class FullyConnectedForImageClassisfication:
    '''
    Simple Fully Connected NN with one hidden layer, for image classification.
    INPUT => FC/RELU => FC/SOFTMAX
    '''
    def build(width, height, depth, hidden_units, classes):
        inputShape = (height, width, depth)
        # Initialize the model
        model = tf.keras.models.Sequential()
        # Flatten the input data
        model.add(tf.keras.layers.Flatten(input_shape=inputShape))
        # FC/RELU layer
        model.add(tf.keras.layers.Dense(hidden_units, activation='relu'))
        # Softmax classifier
        model.add(tf.keras.layers.Dense(classes, activation="softmax"))
        return model


class LeNet5:
    '''
    CNN - Lenet 5.
    INPUT (28px x 28px x 1) => CONV => RELU => POOL => CONV => RELU => POOL => FC => RELU => FC => SOFTMAX
    '''
    def build(width, height, depth, classes):
        # Initialize the model
        model = tf.keras.models.Sequential()
        inputShape = (height, width, depth)
        # C1 Convolutional Layer. Padding='same' gives the same output as input, so the input images could be treated as 32x32 px
        model.add(tf.keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=inputShape, padding='same'))
        # S2 Pooling Layer
        model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        # C3 Convolutional Layer
        model.add(tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        # S4 Pooling Layer
        model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        # C5 Fully Connected Convolutional Layer
        model.add(tf.keras.layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        # Flatten the CNN output so that we can connect it with fully connected layers
        model.add(tf.keras.layers.Flatten())
        # FC6 Fully Connected Layer
        model.add(tf.keras.layers.Dense(84, activation='tanh'))
        # Output Layer
        model.add(tf.keras.layers.Dense(classes, activation='softmax'))
        return model


class SmallerVGGNet:
    '''
    CNN - VGG (smaller than original).
    '''
    def build(width, height, depth, classes):
        model = tf.keras.models.Sequential()
        inputShape = (height, width, depth)
        chanDim = -1 # because we use "channels_last" data format

        # CONV => RELU => POOL
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=inputShape))
        model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))
        model.add(tf.keras.layers.Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=inputShape))
        model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=inputShape))
        model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", input_shape=inputShape))
        model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", input_shape=inputShape))
        model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        # (FC => RELU)
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1024, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))

        # classification with sigmoid activation
        model.add(tf.keras.layers.Dense(classes, activation="sigmoid"))

        return model