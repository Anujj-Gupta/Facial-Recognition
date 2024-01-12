import cv2
import numpy as np
from PIL import Image
import os

recoginze = cv2.face.LBPHFaceRecognizer_create()

path = "DataImages"

def getImageId(path):
    imagePath = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    ids = []
    for imagePaths in imagePath:
        faceImage = Image.open(imagePaths).convert('L')
        facenp = np.array(faceImage)
        id = (os.path.split(imagePaths)[-1].split(".")[1])
        id=int(id)
        faces.append(facenp)
        ids.append(id)
        cv2.imshow("TrainingModel",facenp)
        cv2.waitKey(1)
    return ids,faces


Ids,facedata = getImageId(path)
recoginze.train(facedata,np.array(Ids))
recoginze.write("Trainer.yml")
cv2.destroyAllWindows()
print("Training complete..................................")