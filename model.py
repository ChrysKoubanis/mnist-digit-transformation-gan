from tensorflow.keras import layers, models

def build_next_digit_model():
    
    #Creates a simple Encoder-Decoder Neural Network.
    #Input: Image (28, 28, 1) -> Output: Image (28, 28, 1)
    
    model = models.Sequential()

    # The Encoder compresses the image to understand what the number is
    
    # Layer with 32 filters
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    # Shrink the image by 2 (28x28 -> 14x14)
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    
    # Layer with 64 filters
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    # Shrink the image by 2 (14x14 -> 7x7)
    model.add(layers.MaxPooling2D((2, 2), padding='same'))

    # The Decoder reconstructs the image (upscales it) to draw the next number
    
    #Now we start "drawing"
    # Layer with 64 brushes
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    # Upscale the image (7x7 -> 14x14)
    model.add(layers.UpSampling2D((2, 2)))
    
    # Layer with 32 brushes
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    # Upscale the image(14x14 -> 28x28)
    model.add(layers.UpSampling2D((2, 2)))
    
    # Final Output Layer that turns numbers back into pixels (0 to 1)
    # We use 1 filter because the image is grayscale
    model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

    # Compililng the model with adam optimizer and binary crossentropy (good for black and white images) as the loss function
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    return model

# --- Simple Test Block ---
if __name__ == "__main__":
    model = build_next_digit_model()
    print("Model built successfully!")
    model.summary()