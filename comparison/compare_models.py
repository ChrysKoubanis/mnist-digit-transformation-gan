import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# --- CONFIGURATION ---

# 1. FOR VISUAL COMPARISON (The Images)
# We keep the Regular model here to see the difference
MODEL_FILES = [
    "encoder_decoder_final.keras",   
    "gan_generator_4.keras", 
    "gan_generator_5.keras", 
    "gan_generator_6.keras" 
]
VISUAL_NAMES = ["Regular (Non-GAN)", "Run 4", "Run 5", "Run 6"]

# 2. FOR MATHEMATICAL COMPARISON (The Graph)
# We REMOVED the Regular model as requested, since its loss logic is different
LOG_FILES = [
    "gan_training_log_4.txt",
    "gan_training_log_5.txt",
    "gan_training_log_6.txt"
]
GRAPH_NAMES = ["Run 4", "Run 5", "Run 6"]


def parse_final_loss(log_filename):
    """
    Reads the last 5 lines of a log file and calculates average loss.
    """
    if not os.path.exists(log_filename):
        print(f"Warning: {log_filename} not found.")
        return 0, 0

    try:
        with open(log_filename, 'r') as f:
            lines = f.readlines()
            
        data_lines = [l for l in lines if "|" in l]
        if len(data_lines) == 0: return 0, 0
        
        last_epochs = data_lines[-5:] 
        
        gen_losses = []
        disc_losses = []
        
        for line in last_epochs:
            parts = line.split("|")
            try:
                d_loss = float(parts[1].split(":")[1].strip())
                g_loss = float(parts[2].split(":")[1].strip())
                disc_losses.append(d_loss)
                gen_losses.append(g_loss)
            except:
                pass
            
        return np.mean(disc_losses), np.mean(gen_losses)
        
    except Exception as e:
        print(f"Error parsing {log_filename}: {e}")
        return 0, 0

def plot_mathematical_comparison():
    """
    Generates a Bar Chart comparing ONLY the GAN models.
    """
    print("--- Generating Mathematical Comparison Graph ---")
    
    disc_scores = []
    gen_scores = []
    
    for log_file in LOG_FILES:
        d, g = parse_final_loss(log_file)
        disc_scores.append(d)
        gen_scores.append(g)

    x = np.arange(len(GRAPH_NAMES)) 
    width = 0.35 

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, disc_scores, width, label='Discriminator Loss', color='#ff9999')
    rects2 = ax.bar(x + width/2, gen_scores, width, label='Generator Loss (Accuracy)', color='#66b3ff')

    ax.set_ylabel('Loss Value')
    ax.set_title('Mathematical Performance Comparison (GAN Runs Only)')
    ax.set_xticks(x)
    ax.set_xticklabels(GRAPH_NAMES)
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    fig.tight_layout()
    plt.show()

def compare_visuals():
    """
    Generates the grid of images including the Regular model.
    """
    SAMPLES_PER_DIGIT = 7 
    
    print("\n--- Loading Models for Visual Comparison ---")
    models = []
    for name in MODEL_FILES:
        try:
            print(f"Loading {name}...")
            m = tf.keras.models.load_model(name, compile=False)
            models.append(m)
        except:
            print(f"Error: Could not load {name}. Skipping.")
            models.append(None)

    print("--- Loading Data ---")
    (x_raw, y_raw), (_, _) = mnist.load_data()
    x_raw = x_raw.astype('float32') / 255.
    x_raw = np.reshape(x_raw, (len(x_raw), 28, 28, 1))

    for digit in range(10): 
        print(f"Generating comparison for Digit {digit} -> {(digit+1)%10}...")
        
        indices = np.where(y_raw == digit)[0]
        if len(indices) < SAMPLES_PER_DIGIT: continue
        
        selected_indices = np.random.choice(indices, SAMPLES_PER_DIGIT, replace=False)
        
        # Increased figure height slightly to handle titles better
        fig, axes = plt.subplots(SAMPLES_PER_DIGIT, 5, figsize=(15, 2.2 * SAMPLES_PER_DIGIT))
        
        # Main Title
        fig.suptitle(f"Evolution of Digit {digit} -> {(digit+1)%10}", fontsize=20, y=0.98)

        for row in range(SAMPLES_PER_DIGIT):
            idx = selected_indices[row]
            original_image = x_raw[idx]
            input_tensor = np.expand_dims(original_image, axis=0)
            
            # --- Column 1: Input ---
            ax = axes[row, 0]
            ax.imshow(original_image.reshape(28,28), cmap='gray')
            if row == 0: 
                ax.set_title("Input", fontsize=12, pad=10)
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
                    # Use VISUAL_NAMES list
                    ax.set_title(VISUAL_NAMES[i], fontsize=12, pad=10)
                
                ax.axis('off')

        # This prevents the subplots from overlapping
        plt.tight_layout()
        
        # This adds extra whitespace at the top so the Main Title doesn't hit the "Run 4" labels
        plt.subplots_adjust(top=0.90) 
        
        plt.show() 

if __name__ == "__main__":
    # 1. Graph (GANs only)
    plot_mathematical_comparison()
    
    # 2. Visuals (All Models)
    compare_visuals()