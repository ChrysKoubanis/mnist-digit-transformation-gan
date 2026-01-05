import tensorflow as tf
from tensorflow.keras import layers, models, Input

def build_next_digit_model():
    #U-Net
    inputs = Input(shape=(28, 28, 1))

    # Encoder
    
    # Level 1: 28x28 (Filters increased 32 -> 64)
    c1 = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    
    # Downsample: 28x28 -> 14x14
    p1 = layers.Conv2D(64, (3, 3), strides=2, padding='same')(c1)
    p1 = layers.BatchNormalization()(p1)
    p1 = layers.Activation('relu')(p1)

    # Level 2: 14x14 (Filters increased 64 -> 128)
    c2 = layers.Conv2D(128, (3, 3), padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation('relu')(c2)
    
    # Downsample: 14x14 -> 7x7
    p2 = layers.Conv2D(128, (3, 3), strides=2, padding='same')(c2)
    p2 = layers.BatchNormalization()(p2)
    p2 = layers.Activation('relu')(p2)

    # BOTTLENECK
    # 7x7 (Filters 128 -> 256)
    b = layers.Conv2D(256, (3, 3), padding='same')(p2)
    b = layers.BatchNormalization()(b)
    b = layers.Activation('relu')(b)
    # Dropout here adds noise to force learning
    b = layers.Dropout(0.4)(b) 

    # Decoder

    # Level 2 Decoder: 7x7 -> 14x14
    u1 = layers.UpSampling2D((2, 2))(b)
    merge1 = layers.Concatenate()([u1, c2]) 
    
    d1 = layers.Conv2D(128, (3, 3), padding='same')(merge1)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.Activation('relu')(d1)

    # Level 1 Decoder: 14x14 -> 28x28
    u2 = layers.UpSampling2D((2, 2))(d1)
    merge2 = layers.Concatenate()([u2, c1])
    
    d2 = layers.Conv2D(64, (3, 3), padding='same')(merge2)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.Activation('relu')(d2)

    # Output
    outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d2)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model