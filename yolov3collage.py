from tkinter import filedialog
import tkinter as tk
from PIL import Image, ImageFont, ImageDraw, ImageTk
import cv2
import numpy as np
import time
import os

CONFIDENCE = 0.5 #taux de confiance (50%)
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
nb = 0 # compteur images
blurRate = 45 #taux de floutage
defaultSize = 500 #taille par defaut pour affichage
smallfontSize = 25 #taille de la police pour l'image reduite

root = tk.Tk() # fenetre d'affichage
currentPath = os.getcwd()+"/" # chemin actuel dans lequel se situe le programme
# network configuration
config_path = currentPath+"yolov3.cfg" # organisation du reseau de neurones
# YOLO net weights
weights_path = currentPath+"yolov3.weights" # hierarchie du reseau de neurones
#  coco class labels (objects)
labels = open(currentPath+"coco.names").read().strip().split("\n") # noms des objets detectables

fontFileName=currentPath+'RobotoSlab-Regular.ttf' # police pour affichage sur l'image
loadedFont = ImageFont.truetype(fontFileName,smallfontSize) # chargement de la police pour affichage dans la fenetre
net = cv2.dnn.readNetFromDarknet(config_path, weights_path) # initialisation du reseau de neurones

#fonction pour calculer le ratio de reduction d'une image
def calcRatio(origw, origh):
    ratio = 0
    if(origw >= origh):
        ratio = origw/origh
        nwidth = defaultSize
        nheight = int(nwidth/ratio)
    elif(origh > origw):
        ratio = origh/origw
        nheight = defaultSize
        nwidth = int(nheight/ratio)

    return (ratio,nwidth,nheight)

#fonction de detection
def runDetection(imgFolder,direction):
    global nb # acces au compteur d'images pour modificiation de la valeur
    boxes, confidences, class_ids = [], [], []

    # en arriere ou en avant dans la liste des images (filelist)
    if direction < 0:
        if nb > 1:
            nb-=2
        else:
            nb=len(filelist)-1
    elif direction > 0:
        if(nb >= len(filelist)):
            nb = 0

    imgTarget = imgFolder+"/"+filelist[nb] #le chemin de l'image actuelle a afficher
    print(imgTarget)

    image = cv2.imread(imgTarget) # ouverture de l'image

    if (image is np.ndarray):
        print("EMPTY PIC")
        return

    fullImage = image.copy() # copie de l'image originale (pour besoins de sauvegarde)

    origh, origw = image.shape[:2] #taille originale de l'image

    #calcul du ratio de reduction (pr reduire la plus grne longeur a 500px)
    ratio, nwidth, nheight = calcRatio(origw,origh)

    #reduction de l'image
    image = cv2.resize(image, (nwidth, nheight))

    # tranformation de l'image reduite en "blob" pur traitement par le reseau de neurones
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    image.shape: (1200, 1800, 3) #?
    blob.shape: (1, 3, 416, 416) #?

    # on envoie le blob au reseau de neurones (en input)
    net.setInput(blob)

    #obtention des differentes couches du reseau
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()] # this shit fails on opencv4.5.4 !!!
    layer_outputs = net.forward(ln) # on effectue la detection d'objets

    print("detection done..")

    #pour chaque valeurs retournee par les couches du reseau de neurones..
    for output in layer_outputs:
        #pour chaque detection dans chaque valeur..
        for detection in output:
            print("detection...")
            scores = detection[5:] # informations sur la detection
            class_id = np.argmax(scores) #extraction de l'ID detecte (quel objet?)
            confidence = scores[class_id] #extraction du taux de confiance dans la detection effectuee

            # si le tqux de confiance est superieur au niveau minimum estime (50%)
            if confidence > CONFIDENCE:
                # obtention des coordoonees
                box = detection[:4] * np.array([nwidth, nheight, nwidth, nheight])
                (centerX, centerY, width, height) = box.astype("int") ## obtention des coordoonees du centre de l'objet detecte et de sa taille
                coord_x = int(centerX - (width / 2)) # coordonnee x du point haut gauche
                coord_y = int(centerY - (height / 2)) # coordonnee y du point haut gauche
                boxes.append([coord_x, coord_y, int(width), int(height)]) # ajout des coordoonees et taille de l'objet a la liste des objets
                confidences.append(float(confidence)) # ajout du taux de confiance pour cet objet a a liste des taux de confiance
                class_ids.append(class_id) # ajout de l'id de l'objet detecte a la liste des IDs des objets detectes

    # la fonction suivante supprime d'eventuelles detections multiples d'un meme objet (cf. https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/ ) :
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
    print("shit is done here")
    #si nous avons au moins un objet detecte
    if len(idxs) > 0:
        print("objets detectes")
        # Iterations dans les donnees recuperees (detections) pour dessin et floutage de l'image
        for i in idxs.flatten():
            x, y = boxes[i][0], boxes[i][1] # recuperation des coordonnees de l'objet detecte
            w, h = boxes[i][2], boxes[i][3] # recuperation de la hauteur et de la largeur de l'objet detecte

            #correction necessaire si les valeurs sont negatives:
            if(x < 0):x=0
            if(y < 0):y = 0

            image = cv2.rectangle(image, (x, y), (x + w, y + h), color=(0,0,0), thickness=1) # on dessine un rectangle dans l'image a l'endroit de la detection
            text= f"{labels[class_ids[i]]}" # on prepare le texte (nom de l'objet detecte) pour dessin sur l'image
            image[y+1:y+h, x+1:x+w] = cv2.blur(image[y+1:y+h, x+1:x+w], (blurRate, blurRate)) # on floute le rectangle de la detection

            #preparation au dessin sur l'image originale (taille originale)
            fx = int(x*(origw/nwidth))  #position x de l'objet detecte sur l'image originale
            fy = int(y*(origh/nheight)) #position y de l'objet detecte sur l'image originale
            fw = int(w*(origw/nwidth)) #largeur de l'objet detecte sur l'image originale
            fh = int(h*(origh/nheight)) #longeur de l'objet detecte sur l'image originale

            fullImage = cv2.rectangle(fullImage, (fx, fy), (fx + fw, fy + fh), color=(0,0,0), thickness=1) # dessin d'un rectangle sur l'image originale
            fullBlurRate = int(blurRate*(origw/nwidth)) # mise a l'echelle du taux de floutage
            fullImage[fy+1:fy+fh, fx+1:fx+fw] = cv2.blur(fullImage[fy+1:fy+fh, fx+1:fx+fw], (fullBlurRate, fullBlurRate)) # floutage du rectangle sur l'image originale

            # pour ecrire le texte (label de l'objet detecte) sur les images et avec une police de notre choix
            # nous devons passer par PIL (la police est chargee, pour l'image reduite, en variable globale au debut du programme (loadedFont))
            im_p = Image.fromarray(image) #trasfert de l'image reduite depuis OpenCV (numpy) vers Pil
            fullim_p = Image.fromarray(fullImage) #trasfert de l'image originale depuis OpenCV (numpy) vers Pil

            draw = ImageDraw.Draw(im_p) #ouverture de l'image reduite pour dessin
            fulldraw = ImageDraw.Draw(fullim_p) # ouverture de l'image originale pour dessin

            draw.text((x+2,y-2),text,(0,0,0),font=loadedFont) # pose du texte sur l'image reduite

            scaledFontSize = int(smallfontSize*(origw/nwidth)) # mise a l'echelle de la taille de la police
            FullFont = ImageFont.truetype(fontFileName,scaledFontSize)# chargement de la police a taille augmentee
            fulldraw.text((fx+2,fy-2),text,(0,0,0),font=FullFont)  # pose du texte sur l'image originale

            fullImage=np.array(fullim_p) # conversion depuis Pil vers OpenCV (numpy)
            image = np.array(im_p)# conversion depuis Pil vers OpenCV (numpy)

    # correction des couleurs sur l'image reduite
    b,g,r = cv2.split(image)
    img = cv2.merge((r,g,b))
    imph=Image.fromarray(img) # de nouveau depuis openCV vers Pil

    # correction des couleurs sur l'image originale
    b,g,r = cv2.split(fullImage)
    full = cv2.merge((r,g,b))
    fullImage =Image.fromarray(full) # de nouveau depuis openCV vers Pil

    #affichade de l'image reduite sur la fenetre (GUI)
    ph = ImageTk.PhotoImage(image=imph) # chargement de l'image reduite
    label_image.configure(image=ph, height=nheight, width=nwidth) # configuration du label (espace d'affichage de l'image sur la fenetre)
    label_image.image = ph # image a afficher
    label_image.height = nheight # hauteur de l'image
    label_image.width = nwidth # largeur de l'image
    label_image.place(relx=0.5, rely=0.5, anchor=tk.CENTER) # placement au centre de la fenetre
    buttSave.configure(command=lambda: saveImage(fullImage)) # mise a jour du bouton d'enregistrement de l'image
    buttSave.command=lambda: saveImage(ph) # cliquer sur le bouton fait appel a la foncton saveImage en lui passant l'image floutee de taille originale
    nb+=1 # on prepare l'acces a l'image suivante dans la liste
    print("DONE!")

#cette fonction permet d'enregistrer une image sur le disque dur de l'ordinateur
def saveImage(img):
    filename = filedialog.asksaveasfile(mode='wb', defaultextension=".jpg") # demande a selectioner un fichier ou enregistrer
    if not filename:
         print("no filename")
         return
    img.save(filename) # enregistre


if __name__ == "__main__": # debut du programme
    folder_selected = filedialog.askdirectory() # demande a selectionner un dossier contenant des images
    filelist=os.listdir(folder_selected) # liste les fichiers dans le dossier selectionne

    for fichier in filelist[:]: # ici on retire les fichiers qui ne sont ni jpeg ni png de la liste des fichiers
        if not(fichier.endswith(".jpg")) and not(fichier.endswith(".png")) and not(fichier.endswith(".JPG"))  and not(fichier.endswith(".JPEG")) and not(fichier.endswith(".jpeg")) and not(fichier.endswith(".PNG")):
            filelist.remove(fichier)

    root.geometry("800x800") # on cree la fenetre du programme et on lui donne la taille 800 x 800
    root.configure(bg='black') # on donne a la fenetre un fond noir

    label_image =  tk.Label(root, bg='black') # on cree un Label permettant l'affichage des images
    label_image.pack(side="top", fill="both", expand="yes") # on inscrit le Label dans la fenetre

    #on cree un bouton pour se deplacer en arriere dans la liste des fichiers
    butt =  tk.Button(root, text="<<", command=lambda: runDetection(folder_selected,-1), bg='white')
    butt.pack(side= tk.LEFT) # on positonne le bouton a gauche dans la fenetre

    #on cree un bouton pour se deplacer en avant dans la liste des fichiers
    butt =  tk.Button(root, text=">>", command=lambda: runDetection(folder_selected,1), bg='white')
    butt.pack(side= tk.RIGHT) # on positonne le bouton a droite dans la fenetre

    #on cree un bouton pour enregistrer les images si necessaire
    buttSave =  tk.Button(root,text="save !", bg='white')
    buttSave.pack() # on inscrit le bouton dans la fenetre

    #on lance la detection et l'affichage sur la premiere image de la liste
    runDetection(folder_selected,1)

    root.mainloop() # on maintient l'affichage de la fenetre
