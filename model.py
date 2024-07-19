import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, UpSampling2D

def constructionModel(input_shape=(256, 256, 1)):
    
    entrees = Input(shape=input_shape)

    # Encodeur
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(entrees)              # Sortie (None, 256, 256, 64)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(conv1)     # Sortie (None, 128, 128, 64)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)               # Sortie (None, 128, 128, 128)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(conv3)    # Sortie (None, 64, 64, 128)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)               # Sortie (None, 64, 64, 256)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(conv5)    # Sortie (None, 32, 32, 256)
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)               # Sortie (None, 32, 32, 512)

    # Décodeur
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)               # Sortie (None, 32, 32, 256)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)               # Sortie (None, 32, 32, 128)
    sampling1 = UpSampling2D((2, 2))(conv9)                                             # Sortie (None, 64, 64, 32)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same')(sampling1)           # Sortie (None, 64, 64, 64)
    sampling2 = UpSampling2D((2, 2))(conv10)                                            # Sortie (None, 128, 128, 64)
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(sampling2)           # Sortie (None, 128, 128, 32)
    conv12 = Conv2D(2, (3, 3), activation='tanh', padding='same')(conv11)               # Sortie (None, 128, 128, 2)
    sortie = UpSampling2D((2, 2))(conv12)                                               # Sortie (None, 256, 256, 2)

    return tf.keras.Model(inputs=entrees, outputs=sortie)                               # Retourne le modèle créé
