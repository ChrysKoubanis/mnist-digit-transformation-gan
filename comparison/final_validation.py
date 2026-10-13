import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GAN_MODEL_PATH = os.path.join(SCRIPT_DIR, "gan_generator_6.keras")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
PCA_COMPONENTS = 32 # Similar compression to your 7x7 bottleneck

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (-1, 28, 28, 1))
    x_test = np.reshape(x_test, (-1, 28, 28, 1))
    return (x_train, y_train), (x_test, y_test)

def build_and_train_classifier(x_train, y_train):
    """
    Builds a standard CNN to act as the 'Judge'.
    It learns to recognize real digits.
    """
    print("--- Training Validator (Judge) ---")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train quickly (1 epoch is enough for ~97% accuracy on MNIST)
    model.fit(x_train, y_train, epochs=1, batch_size=64, verbose=1)
    return model

def create_next_digit_pairs(x, y):
    # Same logic as your data_loader
    x_paired = []
    y_expected_labels = [] # We want the label of the NEXT digit
    
    digits = [x[y == i] for i in range(10)]
    
    for n in range(10): 
        target_n = (n + 1) % 10
        inputs = digits[n]
        # We want to check if the output looks like 'target_n'
        
        # Take 100 samples per digit for testing
        samples = inputs[:100]
        x_paired.append(samples)
        y_expected_labels.append(np.full(len(samples), target_n))
        
    return np.concatenate(x_paired), np.concatenate(y_expected_labels)

def run_pca_comparison(x_flat):
    """
    Professor's Requirement: Compare vs PCA Reconstruction.
    """
    print("\n--- Running PCA Comparison ---")
    pca = PCA(n_components=PCA_COMPONENTS)
    
    # 1. Compress (Encode)
    x_compressed = pca.fit_transform(x_flat[:10000]) # Fit on subset
    
    # 2. Decompress (Decode/Reconstruct)
    x_reconstructed = pca.inverse_transform(x_compressed)
    
    # 3. Calculate Loss (MSE)
    mse = np.mean((x_flat[:10000] - x_reconstructed) ** 2)
    print(f"PCA ({PCA_COMPONENTS} components) Reconstruction MSE: {mse:.4f}")
    return x_reconstructed[:5]

def plot_comparison(original, reconstructed, gan_gen=None):
    n = 5  # How many digits to show
    plt.figure(figsize=(10, 4))
    for i in range(n):
        # 1. Original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # 2. PCA Reconstructed
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        plt.title("PCA (Blurry)")
        plt.axis('off')

        # 3. GAN Generated (Optional - if you have the array from the previous step)
        if gan_gen is not None:
            ax = plt.subplot(3, n, i + 1 + 2*n)
            plt.imshow(gan_gen[i].reshape(28, 28), cmap='gray')
            plt.title("GAN (Sharp)")
            plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # 1. Train the Judge (Standard Classifier)
    classifier = build_and_train_classifier(x_train, y_train)
    
    # 2. Prepare GAN Test Data (Input: 3 -> Expected Label: 4)
    x_gan_input, y_expected = create_next_digit_pairs(x_test, y_test)
    
    # 3. Load GAN and Generate
    print(f"\n--- Testing GAN Model: {GAN_MODEL_PATH} ---")
    try:
        gan = tf.keras.models.load_model(GAN_MODEL_PATH, compile=False)
        generated_images = gan.predict(x_gan_input, verbose=0)
        
        # 4. Ask Judge to classify generated images
        predictions = classifier.predict(generated_images, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)
        
        # 5. Calculate Accuracy
        gan_acc = accuracy_score(y_expected, predicted_labels)
        print(f"GAN 'Turing Test' Accuracy: {gan_acc * 100:.2f}%")
        print("(Percentage of generated images recognized correctly by AI)")
        
    except Exception as e:
        print(f"Could not load GAN: {e}")

    # 6. PCA Comparison (Professor's Request)
    # Flatten images for PCA
    x_flat = x_test.reshape(len(x_test), -1)
    run_pca_comparison(x_flat)


# You need to grab the data from the functions to pass here.
# For now, just running the PCA part to see the blur:
(x_train, _), (x_test, _) = load_data()
x_flat = x_test.reshape(len(x_test), -1)
reconstructed_images = run_pca_comparison(x_flat) # This function returns the images!
plot_comparison(x_test[:5], reconstructed_images)

if __name__ == "__main__":
    main()