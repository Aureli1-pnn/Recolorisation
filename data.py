import os
import cv2
import numpy as np 

# Fonction de récupération des images 
def recupData(path, split=0.8):
    x = []
    y = [] 
    nbImages = 0
    for filename in os.listdir(path):
        # Maximum de 6250 images pour les images de célébrité
        if nbImages < 6250:
            # Récupération de l'image couleur non resize
            img = cv2.imread(path + '/' + filename)

            # Passage du format BGR en Lab
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)

            # Resize des images en 256*256 avec 3 Canaux Lab
            img_lab = cv2.resize(img_lab, (256, 256))
            img_lab = img_lab.astype(np.float32)

            # Stockage dans x et y
            x.append(img_lab[:, :, 0] /255)   # Stockage du canal L = gris
            y.append(img_lab[:, :, 1:]/255)   # Stockage des canaux a et b = couleurs

            # Incrément du nombre d'images dans x et y
            nbImages += 1

    # Répartition en train/test
    nbDataTrain = int(len(x)*split)
    x_train = np.array(x[:nbDataTrain])
    x_test  = np.array(x[nbDataTrain:])
    y_train = np.array(y[:nbDataTrain])
    y_test  = np.array(y[nbDataTrain:])

    return x_train, x_test, y_train, y_test

# Fonction de préparation des données
def genererEntrees(x_train, y_train):
    
    nb_donnees = len(x_train)
    for i in range(nb_donnees):
        x_input = x_train[i].reshape(1, 256, 256, 1)
        y_input = y_train[i].reshape(1, 256, 256, 2)
        yield (x_input,y_input)