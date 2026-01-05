import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist

def compare_models():
    # --- 1. CONFIGURATION ---
    model_names = [
        "gan_generator_epoch_10.keras", 
        "gan_generator_epoch_20.keras", 
        "gan_generator_epoch_30.keras", 
        "gan_generator.keras" 
    ]
    
    # How many examples to show per digit?
    SAMPLES_PER_DIGIT = 7 
    
    print("--- Loading Models ---")
    models = []
    for name in model_names:
        try:
            print(f"Loading {name}...")
            m = tf.keras.models.load_model(name, compile=False)
            models.append(m)
        except:
            print(f"Error: Could not load {name}. Skipping.")
            models.append(None)

    print("\n--- Loading Data ---")
    (x_raw, y_raw), (_, _) = mnist.load_data()
    x_raw = x_raw.astype('float32') / 255.
    x_raw = np.reshape(x_raw, (len(x_raw), 28, 28, 1))

    # --- 2. GENERATE FIGURES ---
    
    for digit in range(10): # Loop 0 to 9
        print(f"Generating comparison for Digit {digit} -> {(digit+1)%10}...")
        
        # Find all real examples of this digit
        indices = np.where(y_raw == digit)[0]
        if len(indices) < SAMPLES_PER_DIGIT: continue
        
        # Pick random distinct examples
        selected_indices = np.random.choice(indices, SAMPLES_PER_DIGIT, replace=False)
        
        # Create a figure: Rows = Samples, Columns = Input + 4 Models
        fig, axes = plt.subplots(SAMPLES_PER_DIGIT, 5, figsize=(15, 2 * SAMPLES_PER_DIGIT))
        fig.suptitle(f"Evolution of Digit {digit} -> {(digit+1)%10}", fontsize=16)

        for row in range(SAMPLES_PER_DIGIT):
            idx = selected_indices[row]
            original_image = x_raw[idx]
            input_tensor = np.expand_dims(original_image, axis=0) # Shape (1, 28, 28, 1)
            
            # --- Column 1: Input ---
            ax = axes[row, 0]
            ax.imshow(original_image.reshape(28,28), cmap='gray')
            if row == 0: ax.set_title("Input")
            ax.axis('off')
            
            # --- Columns 2-5: Models ---
            for i, model in enumerate(models):
                ax = axes[row, i+1]
                
                if model is None:
                    ax.text(0.5, 0.5, "Missing", ha='center')
                    ax.axis('off')
                    continue

                prediction = model.predict(input_tensor, verbose=0)
                ax.imshow(prediction[0].reshape(28,28), cmap='gray')
                
                if row == 0:
                    epoch_label = (i+1)*10
                    if i == 3: epoch_label = 40
                    ax.set_title(f"Epoch {epoch_label}")
                
                ax.axis('off')

        plt.tight_layout()
        plt.show() 

if __name__ == "__main__":
    compare_models()