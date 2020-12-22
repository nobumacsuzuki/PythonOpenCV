import cv2
import numpy as np
import sys
import os

def main():
    filename = "lena_512x512.bmp"

    # color
    imageColor = cv2.imread(filename, 1)
    cv2.imshow('Color image',imageColor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # gray
    imageGray = cv2.imread(filename, 0)
    print(imageGray)
    cv2.imshow('Grat image',imageGray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # perspetive transform
    heightImageColor, widthImageColor = imageColor.shape[:2]
    print(heightImageColor)
    print(widthImageColor)
    rectangleSource = np.array([[0, 0], [0, heightImageColor], [widthImageColor, heightImageColor], [widthImageColor, 0]], dtype=np.float32)
    rectangleTarget = np.array([[100, 100], [0, heightImageColor - 100], [widthImageColor, heightImageColor - 100], [widthImageColor - 100, 100]], dtype=np.float32)
    matrixPerspective = cv2.getPerspectiveTransform(rectangleSource, rectangleTarget)
    imagePerspective = cv2.warpPerspective(imageColor, matrixPerspective, (widthImageColor, heightImageColor))
    cv2.imshow('Perspective Transform', imagePerspective)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # face and eye recongnition based on haar cascade
    pathPython  = sys.executable
    pathPython = pathPython.replace('python.exe', '')
    pathCascade = os.path.join(pathPython, 'pkgs\\libopencv-4.0.1-hbb9e17c_0\\Library\\etc\\haarcascades')
    pathCascadeEye = os.path.join(pathCascade, 'haarcascade_eye.xml')
    pathCascadeFrontFace = os.path.join(pathCascade,'haarcascade_frontalface_default.xml')

    face_cascade = cv2.CascadeClassifier(pathCascadeFrontFace)
    eye_cascade = cv2.CascadeClassifier(pathCascadeEye)

    src = cv2.imread(filename)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(src_gray)

    for x, y, w, h in faces:
        cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = src[y: y + h, x: x + w]
        face_gray = src_gray[y: y + h, x: x + w]
        eyes = eye_cascade.detectMultiScale(face_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('face recogntion', src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()