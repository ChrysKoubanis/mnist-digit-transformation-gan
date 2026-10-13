import os

# Must be set BEFORE importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 = FATAL ONLY (Hides info, warnings, and errors)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disables the optimization spam

import warnings
warnings.filterwarnings('ignore') # Hides Python warnings

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Import our modular parts
from data_loader import load_next_digit_data
from model import build_next_digit_model  # The Generator
from discriminator import build_discriminator # The Discriminator

# Configuration
BATCH_SIZE = 64 # Smaller batch size often helps GANs
EPOCHS = 40
LAMBDA = 10    # How much we care about the "Correct Number" vs "Realistic Looking"

def train_gan():
    # Setup
    print("Loading Data...")
    (x_train, y_train), (x_test, y_test) = load_next_digit_data()
    
    # Convert to efficient TensorFlow Dataset (Makes training faster)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

    # Build Models
    print("\nBuilding GAN Models....")
    generator = build_next_digit_model()
    discriminator = build_discriminator()

    # Optimizers (The logic that updates the weights)
    gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    # Loss Functions
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    mae = tf.keras.losses.MeanAbsoluteError()

    # Custom training step
    # This function runs for one batch of data
    @tf.function 
    def train_step(input_image, target_image):
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate Fake Images
            generated_image = generator(input_image, training=True)

            # Let Discriminator Judge
            real_output = discriminator(target_image, training=True)
            fake_output = discriminator(generated_image, training=True)

            # Calculate Discriminator Loss
            # Label Smoothing (Use 0.9 instead of 1.0 for Real)
            # This prevents the discriminator from being too confident
            smooth_real_labels = tf.ones_like(real_output) * 0.9 
            
            disc_loss_real = cross_entropy(smooth_real_labels, real_output)
            disc_loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
            disc_loss = disc_loss_real + disc_loss_fake

            # Generator Loss
            # GAN Loss: Trick it to predict 1.0
            gen_gan_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
            # Pixel Loss
            gen_l1_loss = mae(target_image, generated_image)
            
            gen_total_loss = gen_gan_loss + (LAMBDA * gen_l1_loss)

        # E. Update Gradients
        gradients_of_generator = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_total_loss, disc_loss

    # main training loop
    print("\nStarting GAN Training...")
    
    # Open a text file to save the history
    with open("gan_training_log.txt", "w") as log_file:
        log_file.write("Epoch, Disc_Loss, Gen_Loss, Time_Sec\n") # Header
        
        for epoch in range(EPOCHS):
            start = time.time()
            
            # Train on every batch
            for input_batch, target_batch in train_dataset:
                gen_loss, disc_loss = train_step(input_batch, target_batch)

            # Format the output string
            stats = f'Epoch {epoch + 1}/{EPOCHS} | Disc Loss: {disc_loss:.4f} | Gen Loss: {gen_loss:.4f} | Time: {time.time()-start:.1f}s'
            
            # Print to Console
            print(stats)
            
            # Write to file and save
            log_file.write(stats + "\n")
            log_file.flush() 

            # Visualize progress every 10 epochs (Saves image to file)
            if (epoch + 1) % 10 == 0:
                # Save the Model (with specific filename for this epoch)
                checkpoint_name = f'gan_generator_epoch_{epoch + 1}.keras'
                generator.save(checkpoint_name)
                print(f"   Saved model: {checkpoint_name}")

                # Save the Visualization (Photo)
                # The function handles the saving if we pass epoch_number
                show_predictions(generator, x_test, y_test, epoch_number=epoch + 1)

    # Save the specialized generator
    generator.save('gan_generator.keras')
    print("Training Complete. Model saved.")
    
    # Final Show
    show_predictions(generator, x_test, y_test)

def show_predictions(model, x, y, epoch_number=None):
    # Simple visualizer
    num_samples = 10
    indices = np.random.randint(0, len(x), num_samples)
    
    preds = model.predict(x[indices], verbose=0)
    
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        # Input
        plt.subplot(3, num_samples, i + 1)
        plt.imshow(x[indices][i].reshape(28,28), cmap='gray')
        plt.axis('off')
        if i==4: plt.title("Input (3)")

        # Target
        plt.subplot(3, num_samples, i + 1 + num_samples)
        plt.imshow(y[indices][i].reshape(28,28), cmap='gray')
        plt.axis('off')
        if i==4: plt.title("Real Target (4)")

        # GAN Output
        plt.subplot(3, num_samples, i + 1 + num_samples*2)
        plt.imshow(preds[i].reshape(28,28), cmap='gray')
        plt.axis('off')
        if i==4: plt.title("GAN Output")
        
    plt.tight_layout()

    if epoch_number is not None:
        filename = f"gan_progress_epoch_{epoch_number}.png"
        plt.savefig(filename)
        print(f"Saved visualization to {filename}")
        plt.close() # Closes the plot memory so training continues
    else:
        plt.show() # Keep standard show for the very last run if you want

if __name__ == "__main__":
    train_gan()