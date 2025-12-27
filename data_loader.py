import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

def load_next_digit_data():
    
    #Loads MNIST and creates pairs:
    #X = Image of Digit N
    #Y = Image of Digit N+1 (e.g., Input 3 -> Target 4)
    
    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data() # Contains 70,000 images of handwritten digits

    # Normalize from [0, 255] pixels to [0, 1] and reshape to (28, 28, 1) because the images are 28x28 and 1 color channel
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

    def create_pairs(x, y):
        x_paired = []
        y_paired = []
        
        # Group images by their digit (0, 1, 2...)
        digits = [x[y == i] for i in range(10)]
        
        # Create pairs: 0->1, 1->2, ...
        for n in range(10): 
            target_n = (n + 1) % 10  # so that when we get 9 the target is 0
            
            inputs = digits[n]
            targets = digits[target_n]
            
            # We need equal amounts, so trim to the smaller count
            min_len = min(len(inputs), len(targets))
            
            x_paired.append(inputs[:min_len])
            y_paired.append(targets[:min_len])
            
        return np.concatenate(x_paired), np.concatenate(y_paired) # "stitches" all the lists together

    print("Creating Next-Digit pairs (e.g., 3 -> 4)...")
    x_train_next, y_train_next = create_pairs(x_train, y_train)
    x_test_next, y_test_next = create_pairs(x_test, y_test)
    
    # Shuffle the data (important for training)
    indices = np.arange(x_train_next.shape[0])
    np.random.shuffle(indices)
    x_train_next = x_train_next[indices]
    y_train_next = y_train_next[indices]
    
    print("Data ready. Training pairs:", x_train_next.shape[0])
    return (x_train_next, y_train_next), (x_test_next, y_test_next)

# This block allows us to run the file directly to test it
if __name__ == "__main__":
    (x_tr, y_tr), _ = load_next_digit_data()
    
    print("Displaying 5 random pairs to verify logic...")
    plt.figure(figsize=(10, 4))
    for i in range(5):
        # Input
        ax = plt.subplot(2, 5, i + 1)
        plt.imshow(x_tr[i].reshape(28, 28), cmap='gray')
        plt.title("Input")
        plt.axis('off')
        
        # Target
        ax = plt.subplot(2, 5, i + 1 + 5)
        plt.imshow(y_tr[i].reshape(28, 28), cmap='gray')
        plt.title("Target")
        plt.axis('off')
    plt.show()

#This script does not recognize what a number is, nor does it generate new pixels. 
#It simply organizes real human written photos into "Flashcards" so a future Neural Network 
#can learn the transformation from one number to the next