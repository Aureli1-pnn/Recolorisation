import tensorflow as tf
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import os 

from tensorflow.keras.models import load_model
from tkinter.filedialog import askopenfilename
from PIL import Image

# Retourne une image en RGB 256*256*3 à partir d'un chemin
def imageRGB(path_img):
    img = cv2.imread(path_img)  # Chargement de l'image au format BGR
    img = cv2.resize(img, (256, 256)) # Resize en 256*256*3

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
# Retourne l'image en noir et blanc sous la forme d'un tableau au format 256*256*1
def preparerImage(path_img):
    img_bgr = cv2.imread(path_img)  # Chargement de l'image au format BGR

    # Passage du format BGR en Lab
    img_rgb          = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # BGR -> RGB
    img_lab_original = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab) # RGB -> LAB
    img_lab_original = cv2.resize(img_lab_original, (256, 256)) # Resize en 256*256*3

    # Normalisation des pixels
    img_lab = img_lab_original.astype(np.float32)   
    x = np.array(img_lab[:, :, 0] /255)             

    return x.reshape(1, 256, 256, 1)

# Fonction retournant la prédiction d'un model au format 256*256*2
def prediction(model, x):
    y_predict = model.predict(x)
    y_predict = y_predict*255

    return y_predict.reshape(256, 256, 2)

# Fonction qui reconstruit l'image à partir de la prédiction au format RGB 256*256*3
def reconstruction(path_img, y_predict):
    img_bgr = cv2.imread(path_img)  # Chargement de l'image au format BGR

    # Passage du format BGR en Lab
    img_rgb          = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # BGR -> RGB
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab) # RGB -> LAB
    img = cv2.resize(img, (256, 256)) # Resize en 256*256*3
    img[:, :, 1:] = y_predict # Remplace les deux canaux A et B par la prédiction

    return cv2.cvtColor(img, cv2.COLOR_Lab2RGB)

# Fonction qui transforme une image BGR en niveau de gris
def transformationNoirEtBlanc(path_img):
    img = cv2.imread(path_img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return cv2.resize(img, (256, 256))

# Fonction prédisant et affichant une image fournit par l'utilisateur
def predictAndShow(model):
    img_path = askopenfilename(initialdir= "./", title="Selectionner une image", filetypes=[("Fichier jpeg", "*.jpg")])
    x = preparerImage(img_path)

    # Prédiction
    y_predict = prediction(model, x)

    # Reconstruction de l'Image
    image_predict = reconstruction(img_path, y_predict)

    # Affichage de l'image originale
    plt.figure(figsize=(30,20), frameon=False)
    plt.subplot(5,5,1)
    plt.title("Image originale")
    plt.imshow(imageRGB(img_path))

    # Affichage de l'image en noir et blanc
    plt.subplot(5,5,2)
    plt.title("Image noir et blanc")
    plt.imshow(transformationNoirEtBlanc(img_path))

    # Affichage de l'image recoloriser 
    plt.subplot(5,5,3)
    plt.title("Image prédite")
    plt.imshow(image_predict)

    plt.waitforbuttonpress() # Attente avant de fermer la fenêtre
    
    # Enregistrement de l'image dans result
    basename = os.path.basename(img_path)
    plt.clf()
    if not os.path.exists("./Images/result"):
        os.makedirs("./Images/result")
    plt.imshow(image_predict)
    plt.savefig("./Images/result/" + basename)

############## Main ################
model = load_model("model_final/model_final.h5")
continuer = True
while(continuer):
    predictAndShow(model)
    choix = input("Voulez vous continuer 1 pour oui sinon n'importe quoi d'autre : ")
    if int(choix) != 1:
        continuer=False
print("\nFin du programme ! Vous pouvez trouver vos résultats dans le dossier result.")