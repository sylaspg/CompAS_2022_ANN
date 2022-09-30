# Based on https://www.pyimagesearch.com/2020/03/30/autoencoders-for-content-based-image-retrieval-with-keras-and-tensorflow/

# Imports
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np


class CNNAutoencoder:
    ''' Example autoencoder based on CNN network '''

    def build(width, height, depth, filters=(32, 64), latentDim=16):
        # Initialize the input shape to be "channels last" along with channels dimension
        inputShape = (height, width, depth)
        chanDim = -1

        # Define the input to the encoder
        inputs = Input(shape=inputShape)
        x = inputs

        # Loop over the number of filters
        for f in filters:
            # CONV => RELU => BN
            x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chanDim)(x)

        # Flatten the network and construct the latent vector
        volumeSize = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(latentDim, name="encoded")(x)

        # Start building the decoder, with inputs as output from the encoder
        x = Dense(np.prod(volumeSize[1:]))(latent)
        x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

        # Loop over filters again, but this time in reverse order
        for f in filters[::-1]:
            # CONV_TRANSPOSE => RELU => BN
            x = Conv2DTranspose(f, (3, 3), strides=2,
                padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chanDim)(x)

        # Single CONV_TRANSPOSE layer used to recover the original depth of the image
        x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
        outputs = Activation("sigmoid", name="decoded")(x)

        # Construct and return the autoencoder
        autoencoder = Model(inputs, outputs, name="autoencoder")
        return autoencoder
        
        
def visualize_predictions(orig, decoded, samples=10):
    ''' Visualize predictions - original and predicted images side by side '''

    # Initialize list of output images
    outputs = None

    # Loop over number of output samples
    for i in range(0, samples):
        # Grab the original image and reconstructed image
        original = (orig[i]*255).astype("uint8")
        reconstructed = (decoded[i]*255).astype("uint8")

        # Stack the original and reconstructed image side-by-side
        output = np.hstack([original, reconstructed])

        # If the outputs array is empty, initialize it as the current side-by-side image display
        if outputs is None:
            outputs = output
        else: # Otherwise, vertically stack the outputs
            outputs = np.vstack([outputs, output])

    return outputs
