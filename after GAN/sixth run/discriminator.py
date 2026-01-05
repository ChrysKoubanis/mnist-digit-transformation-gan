from tensorflow.keras import layers, models, Input

def build_discriminator():
    
    #PatchGAN version
    #Input: An Image (28, 28, 1)
    #Output: A 7x7 Grid of scores (Each pixel in the grid votes on a region)
    
    model = models.Sequential()
    
    model.add(Input(shape=(28, 28, 1)))

    model.add(layers.Conv2D(64, (3, 3), strides=2, padding='same'))
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (3, 3), strides=2, padding='same'))
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Dropout(0.3))

    #instead of flattening, we keep it as an image.
    #This outputs a 7x7 grid where 1=Real and 0=Fake.
    model.add(layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid'))

    return model