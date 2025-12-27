import os
import warnings
# --- SUPPRESS LOGS ---
# 0 = all messages, 1 = no info, 2 = no warnings, 3 = no errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- SUPPRESS PYTHON WARNINGS ---
# This ignores the "np.object" FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_next_digit_data
from model import build_next_digit_model

def train_network():
    print("Loading Data ...")
    (x_train, y_train), (x_test, y_test) = load_next_digit_data()

    # Initialize the Model
    print("\nBuilding Model ...")
    model = build_next_digit_model() #  We start with a model with random weights that doesnt "know" anything

    # Start the training
    print("\nStarting Training ...")

    history = model.fit(  
        x_train, y_train,
        epochs=20,            # 20 repetitions
        batch_size=128,       # 128 images each epoch
        shuffle=True,
        validation_data=(x_test, y_test), # Checks its score on the test data
        verbose=1             # Keeps the progress bar, hides extra noise
    )

    # Save the Model
    print("\nSaving Model ...")
    model.save('next_digit_generator.keras') # Python automatically overwrites the existing file after every run ,
                                             #it deletes the old one and replaces it with the new one instantly
    print("Model saved as 'next_digit_generator.keras'")

    # Visual Test
    print("\nVisualizing Predictions ...")
    
    # Let the model guess on 10 random test images
    num_samples = 10
    indices = np.random.randint(0, len(x_test), num_samples)
    
    test_inputs = x_test[indices]
    test_targets = y_test[indices]

    # Get the blurry predictions
    raw_predictions = model.predict(test_inputs, verbose=0)

    # Contrast technique
    # We boost contrast by squaring the values.
    # This pushes gray to black but keeps white white.
    sharpened_predictions = raw_predictions ** 2

    plt.figure(figsize=(20, 6))
    for i in range(num_samples):
        # Row 1: Input
        ax = plt.subplot(3, num_samples, i + 1)
        plt.imshow(test_inputs[i].reshape(28, 28), cmap='gray')
        plt.title("Input")
        plt.axis('off')

        # Row 2: Real Target
        ax = plt.subplot(3, num_samples, i + 1 + num_samples)
        plt.imshow(test_targets[i].reshape(28, 28), cmap='gray')
        plt.title("Real Target")
        plt.axis('off')

        # Row 3: AI Generated
        ax = plt.subplot(3, num_samples, i + 1 + num_samples * 2)
        plt.imshow(sharpened_predictions[i].reshape(28, 28), cmap='gray')
        plt.title("AI Generated")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_network()


# First run: the model output is blurry because the AI averages all the different ways the n+1 number is written in the images we have given it and
# the average of the different ways to write any number looks blurry.
# To fix this i will change the loss function to mae (Mean Absolute Error) and increase the epochs from 10 to 20
# This punishes small gray errors less but punishes big errors more, encouraging the model to be bolder and sharper rather than safe and blurry.

# Second run all the model output images are black, this phenomenon is called "Collapsing to the Background". 
# It happened because 90% of the pixels in the images are black (the background) and the rest 10% (the number) are white and
# the model realized that if what it guesses is black for every pixel, it is right 90% 0f the time and its error score is low because it only missed the white line.
# As a result the model took the safe path and painted everything black.
# To fix that we need a loss function that punishes the AI more for missing the white pixels than for getting the black background right.
# We will try the mse (Mean Squared Error) loss function.It is sharper than the original binary_crossentropy ,but it punishes "lazy" errors more than mae,
# so the model will be less likely to output pure black.If that also doesnt work i will use a technique called Weighted Loss to increase the significance of the white pixels.

# Third run all the model output images are black again, i will try the Weighted loss mse function and it will hopefully work.
# Also the loss stopped moving entirely for 19 epochs and stayed at 0.1127.This confirms that the model had indeed "collapsed"  and was just predicting black.
# 

# Fourth run all the model output images where very blurry, more blurred than the binary crossentropy way.
# I decided to use the binary crossentropy loss function again and add a sharpness filter that gets the blurry image and
# if a pixel is > 0.5 it becomes white and if it is < 0.5 it becomes black.

# Fifth run some output images look ok (like some 7s and some 3s) ,but the rest look fragmented because the model is under-confident.
# To fix that we need to lower the sharpness filter and accept drak grey lines as white.I changed the filter from 0.5 to 0.2 and also added batch normalization 
# to normalize the signals between layers

# Sixth run most of the output images look blocky and lego-like.This is probably happening because of the way i chose to upscale the images.
# In the Decoder ,Upsampling2D takes a pixel and duplicates it to make it look bigger.When i applied the threshold filter these blocks became very obvious.
# To fix that i will use smart upscaling.I swapped all UpSampling2D for Conv2DTranspose.
# I used stride=2 instead of MaxPooling.That lets the model learn the best way to shrink the image.

# Seventh run small difference but a bit better than before.I will change the sharpness filter, because the current sharpness filter forbids any gray pixels to exist at the edges.
# Instead of that i will use a technique to boost contrast by squaring the values.
# This pushes gray color to black but keeps white white.

# Eighth run, the output looks way better for all numbers excluding one instance.The output images are readable but blurry.
# I dont think we can progress much if i tweak the amount of layers or change the activation function or do different contrast tricks.
# The core issue is that the model sees a 3 and understands that it needs to draw a 4, but doesnt know which one (small one? large one? thick one? a flat one? ....).
# So it minimizes its error by drawing the average of all possible 4s and  the result is blurry.


# While this model fulfills the assignment requirements, the output remains blurry.
# This is due to the "Regression to the Mean" problem: when the model is unsure which specific style 
# of '4' to draw, it minimizes error by drawing the average of all possible '4's.

# To solve this, I plan to implement a Conditional GAN (Generative Adversarial Network).
# A GAN consists of two competing models:

# The Generator: 
#    Takes the input number and tries to create a realistic image of the next number.
#    Its goal is to create images so good that the Discriminator classifies them as "Real".

# The Discriminator: 
#    It is trained on two types of data:
#        a) Real images from the dataset (Label: 1 / Real)
#        b) Fake images from the Generator (Label: 0 / Fake)
#    - Its goal is to correctly distinguish between real human handwriting and the AI's creation.

# The Training Process:
# As the Discriminator gets better at spotting fakes, the Generator is forced to stop creating 
# blurry averages and start creating sharp, specific strokes to fool the Discriminator.
# This adversarial process eventually lead to realistic looking images of digits.

# Challenges:
#   Instability: GANs are notoriously hard to train. If the Discriminator learns too fast, 
#   the Generator collapses. If it learns too slow, the Generator learns nothing.

#   Training Time: Usually, GANs take days to train on high-resolution images.
#   However, since MNIST images are tiny (28x28) and simple (grayscale [0,1]), 
#   the computational load is very low. The "search space" for the Generator is small, meaning 
#   it can learn to fool the Discriminator much faster than it would with color photos. 
#   This makes it a perfect candidate for a high-performance, low-latency GAN implementation.