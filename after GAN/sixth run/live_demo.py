import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Clean output
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist

def run_demo():
    print("--- Loading Saved Model ---")
    try:
        # Load the trained Generator
        model = tf.keras.models.load_model('gan_generator.keras', compile=False)
        print("Model loaded successfully!")
    except:
        print("Error: Could not find 'gan_generator.keras'. Did you finish training?")
        return

    print("--- Loading Real Images ---")
    # We load raw MNIST data so we can pick specific numbers
    (x_raw, y_raw), (_, _) = mnist.load_data()
    x_raw = x_raw.astype('float32') / 255.
    x_raw = np.reshape(x_raw, (len(x_raw), 28, 28, 1))

    while True:
        print("\n" + "="*40)
        user_input = input("Enter a number (0-9) to generate the next digit (or 'q' to quit): ")
        
        if user_input.lower() == 'q':
            break
            
        try:
            digit = int(user_input)
            if digit < 0 or digit > 9:
                print("Please enter a digit between 0 and 9.")
                continue
                
            # 1. Find all examples of this digit in the dataset
            indices = np.where(y_raw == digit)[0]
            
            # 2. Pick one random example
            idx = np.random.choice(indices)
            input_image = x_raw[idx]
            
            # 3. Add batch dimension (1, 28, 28, 1) because model expects a batch
            input_tensor = np.expand_dims(input_image, axis=0)
            
            # 4. LIVE PREDICTION
            print("Generating...")
            prediction = model.predict(input_tensor, verbose=0)
            
            # 5. Show results
            plt.figure(figsize=(8, 4))
            
            # Left: What you gave it
            plt.subplot(1, 2, 1)
            plt.imshow(input_image.reshape(28,28), cmap='gray')
            plt.title(f"Input: {digit}")
            plt.axis('off')
            
            # Right: What it created
            plt.subplot(1, 2, 2)
            plt.imshow(prediction[0].reshape(28,28), cmap='gray')
            plt.title(f"AI Generated: {(digit + 1)%10}")
            plt.axis('off')
            
            print(f"Displaying result: {digit} -> {(digit + 1)%10}")
            plt.show()
            
        except ValueError:
            print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    run_demo()