import tensorflow as tf 
from tensorflow.keras import layers, models ,Input

def build_next_digit_model():
    
    #Creates a simple Encoder-Decoder Neural Network.
    #Input: Image (28, 28, 1) -> Output: Image (28, 28, 1)
    
    model = models.Sequential()

    # The Encoder compresses the image to understand what the number is

    # Explicitly define input to silence warnings
    model.add(Input(shape=(28, 28, 1)))
    
    # Layer with 32 filters
    model.add(layers.Conv2D(32, (3, 3), strides=2, padding='same'))
    model.add(layers.BatchNormalization())  
    model.add(layers.Activation('relu'))
    
    
    # Layer with 64 filters
    model.add(layers.Conv2D(64, (3, 3), strides=2, padding='same'))
    model.add(layers.BatchNormalization()) 
    model.add(layers.Activation('relu'))


    # Here we dont want to use an output layer for the encoder that has 10 outputs (one for each number),
    # because if we do that we will lose all the details (the 7*7*64=3136 numbers) that we want to keep to generate the next number image.
    # If this was a classifier problem we would use the 10 neuron layer.

    # The Decoder reconstructs the image (upscales it) to draw the next number
    
    # First reconstruction layer with 64 filters
    model.add(layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
   
    
    # Refine features with 32 filters
    model.add(layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    # Final Output Layer that turns numbers back into pixels (We use the sigmoid function so the numbers are between 0 and 1)
    # We use 1 filter because the image is grayscale
    model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

    # Compililng the model with adam optimizer and binary crossentropy (good for black and white images) as the loss function
    # model.compile(optimizer='adam', loss='binary_crossentropy')
    # After the first run i dicided to change to loss funciton to mean absolute error to "fix" the bluryness
    #model.compile(optimizer='adam', loss='mae')

    # After the second run the output was black so i changed the loss function to mse.
    #model.compile(optimizer='adam', loss='mse')

    #After the 3rd run the output was black again ,so now i will try a custom weighted loss function.
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

if __name__ == "__main__":
    model = build_next_digit_model()
    print("Model built successfully!")
    model.summary()