import cv2
import numpy as np
import sys
import os
from matplotlib import pyplot as plt

def main():
    filename = "lena_512x512.bmp"

    # color
    imageColor = cv2.imread(filename, cv2.IMREAD_COLOR)
    cv2.imshow('Color image',imageColor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # gray
    imageGray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    print(imageGray)
    cv2.imshow('Grat image',imageGray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # perspetive transform
    offsetPerspective = 100
    heightImageColor, widthImageColor = imageColor.shape[:2]
    print(heightImageColor)
    print(widthImageColor)
    rectangleSource = np.array([[0, 0], [0, heightImageColor], [widthImageColor, heightImageColor], [widthImageColor, 0]], dtype=np.float32)
    rectangleTarget = np.array([[offsetPerspective, offsetPerspective], [0, heightImageColor - offsetPerspective], [widthImageColor, heightImageColor - offsetPerspective], [widthImageColor - offsetPerspective, offsetPerspective]], dtype=np.float32)
    matrixPerspective = cv2.getPerspectiveTransform(rectangleSource, rectangleTarget)
    imagePerspective = cv2.warpPerspective(imageColor, matrixPerspective, (widthImageColor, heightImageColor))
    cv2.imshow('Perspective Transform', imagePerspective)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # face and eye recongnition based on haar cascade
    pathPython  = sys.executable
    pathPython = pathPython.replace('python.exe', '')
    pathCascade = os.path.join(pathPython, r'pkgs\libopencv-4.0.1-hbb9e17c_0\Library\etc\haarcascades')
    pathCascadeEye = os.path.join(pathCascade, 'haarcascade_eye.xml')
    pathCascadeFrontFace = os.path.join(pathCascade,'haarcascade_frontalface_default.xml')

    cascadeClassifierFace = cv2.CascadeClassifier(pathCascadeFrontFace)
    cascadeClassifierEye = cv2.CascadeClassifier(pathCascadeEye)

    faces = cascadeClassifierFace.detectMultiScale(imageGray) # it returns the list of ([tartX, startY, width, height]
    print(faces)

    colorFace = (255, 0, 0) # color is tuple in CV
    colorEye = (0, 255, 0)
    LineThickness = 2

    for (facePositionX, facePositionY, faceWidth, faceHeight) in faces:
        cv2.rectangle(imageColor, (facePositionX, facePositionY), (facePositionX + faceWidth, facePositionY + faceHeight), colorFace, LineThickness)
        bufferFace = imageColor[facePositionY: facePositionY + faceHeight, facePositionX: facePositionX + faceWidth]
        bufferFaceGray = imageGray[facePositionY: facePositionY + faceHeight, facePositionX: facePositionX + faceWidth]
        eyes = cascadeClassifierEye.detectMultiScale(bufferFaceGray)
        print(eyes)
        for (eyePositionX, eyePositionY, eyeWidth, eyeHeight) in eyes:
            cv2.rectangle(bufferFace, (eyePositionX, eyePositionY), (eyePositionX + eyeWidth, eyePositionY + eyeHeight), colorEye, LineThickness)

    cv2.imshow('face recogntion', imageColor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # histogram
    imageColor = cv2.imread(filename, cv2.IMREAD_COLOR)
    color = ('b','g','r')
    print(color)
    for i,col in enumerate(color):
        histgramLena = cv2.calcHist([imageColor],[i],None,[256],[0,256])
        plt.plot(histgramLena, color = col)
        plt.xlim([0,256])
    plt.show()

if __name__ == "__main__":
    main()