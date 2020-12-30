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

    cascadeClassifierFace = cv2.CascadeClassifier(pathCascadeFrontFace) # it instatiates cascade classifier object
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
    colors = ('b','g','r') #matlibplot has the color index, https://matplotlib.org/3.1.0/api/colors_api.html#module-matplotlib.colors
    for (indexColor, color) in enumerate(colors): #enumerate provides list of object with index, in this case [(0, 'b'), (1, 'g'), (2, 'r')] 
        # https://docs.opencv.org/master/d1/db7/tutorial_py_histogram_begins.html
        # cv.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
        # images : it is the source image of type uint8 or float32. it should be given in square brackets, ie, "[img]".
        # channels : it is also given in square brackets. It is the index of channel for which we calculate histogram. For example, if input is grayscale image, its value is [0]. For color image, you can pass [0], [1] or [2] to calculate histogram of blue, green or red channel respectively.
        # mask : mask image. To find histogram of full image, it is given as "None". But if you want to find histogram of particular region of image, you have to create a mask image for that and give it as mask. (I will show an example later.)
        # histSize : this represents our BIN count. Need to be given in square brackets. For full scale, we pass [256].
        # ranges : this is our RANGE. Normally, it is [0,256].
        bufferHistogram = cv2.calcHist([imageColor],[indexColor],None,[256],[0,256]) # it retunrs list of channel histogram
        plt.plot(bufferHistogram, color = color)
        plt.xlim([0,256])
    plt.show()

    ndarrayImageGray = imageGray
    print(ndarrayImageGray)
    print(ndarrayImageGray.dtype)
    print(ndarrayImageGray.shape)
    cv2.imshow('Y channel only', ndarrayImageGray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # FFT and centered
    arrayFFTed = np.fft.fft2(ndarrayImageGray)
    arrayShiftedFFTed = np.fft.fftshift(arrayFFTed)

    # Frequency component power spectrum
    arrayPowerSpectrumFFTRealPart = 20 * np.log(np.absolute(arrayShiftedFFTed))

    ndarrayBuffer = np.copy(arrayPowerSpectrumFFTRealPart)
    min = np.min(ndarrayBuffer)
    ndarrayBuffer[:,:] -= min
    max = np.max(ndarrayBuffer)
    ndarrayBuffer[:,:] /= max
    ndarrayBuffer[:,:] *= 255
    print(ndarrayBuffer)
    ndarrayBufferUint8 = ndarrayBuffer.astype(np.uint8)
    print(ndarrayBufferUint8.dtype)
    print(ndarrayBufferUint8.shape)
    cv2.imshow('PowerSpectrum', ndarrayBufferUint8)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Uncentered and Inverse FFT
    arrayRevertedShiftedFFTed = np.fft.fftshift(arrayShiftedFFTed)
    arrayInvertFFTed = np.fft.ifft2(arrayRevertedShiftedFFTed).real
    ndarrayBufferUint8 = arrayInvertFFTed.astype(np.uint8)
    print(ndarrayBufferUint8.dtype)
    print(ndarrayBufferUint8.shape)

    cv2.imshow('Revered Back Image', ndarrayBufferUint8)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()