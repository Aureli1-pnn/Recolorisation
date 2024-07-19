import tensorflow as tf
import numpy as np 
import cv2
import os
import warnings

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint

from data import recupData, genererEntrees

from model import constructionModel

################ Système d'affichage ##################
warnings.filterwarnings("ignore")
def cls():
    os.system('cls' if os.name=='nt' else 'clear')
cls()

# Créer un callback de sauvegarde pour l'entraînement #
checkpoint_path = "model/cp-{epoch:02d}.ckpt"

checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_freq='epoch')

############### Récupération des données ###############
print("Récupération des données en cours")
path  = './Images/img_align_celeba'
split = 0.8

x_train, x_test, y_train, y_test = recupData(path, split)

##################### Entraînement #####################
cls()
print("Entraînement en cours")
shape    = (256, 256, 1)
types    = (tf.float32, tf.float32)
ds_train = tf.data.Dataset.from_generator(lambda: genererEntrees(x_train, y_train), types).shuffle(100).repeat()

model = constructionModel(shape)
model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
model.summary()
model.fit(ds_train, epochs=5, steps_per_epoch=5000, callbacks=[checkpoint_callback])
model.save("model/model.h5")

###################### Evaluation ######################
print("\nEvaluation en cours...")
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss: ", score)