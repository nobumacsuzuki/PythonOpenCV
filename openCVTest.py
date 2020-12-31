import cv2
import numpy as np
import sys
import os
from matplotlib import pyplot as plt

def main():
    filename = "lena_512x512.bmp"

    # color
    arrayImageColor = cv2.imread(filename, cv2.IMREAD_COLOR) ## cv2 returns ndarray of BGR orientation
    print(arrayImageColor.dtype)
    print(arrayImageColor.shape)
    cv2.imshow('Color image',arrayImageColor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # convert to gray
    arrayImageGray = cv2.cvtColor(arrayImageColor, cv2.COLOR_BGR2GRAY)
    print(arrayImageGray.dtype)
    print(arrayImageGray.shape)
    cv2.imshow('Gray image',arrayImageGray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # perspetive transform
    offsetPerspective = 100
    (heightImageColor, widthImageColor) = arrayImageColor.shape[:2]
    print(heightImageColor)
    print(widthImageColor)
    arrayRectangleSource = np.array([[0, 0], [0, heightImageColor], [widthImageColor, heightImageColor], [widthImageColor, 0]], dtype=np.float32)
    arrayRectangleTarget = np.array([[offsetPerspective, offsetPerspective], [0, heightImageColor - offsetPerspective], [widthImageColor, heightImageColor - offsetPerspective], [widthImageColor - offsetPerspective, offsetPerspective]], dtype=np.float32)
    arrayMatrixPerspective = cv2.getPerspectiveTransform(arrayRectangleSource, arrayRectangleTarget)
    arrrayImagePerspective = cv2.warpPerspective(arrayImageColor, arrayMatrixPerspective, (widthImageColor, heightImageColor))
    cv2.imshow('Perspective Transform', arrrayImagePerspective)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    arrayImageGray = arrayImageGray
    print(arrayImageGray)
    print(arrayImageGray.dtype)
    print(arrayImageGray.shape)
    cv2.imshow('Gray', arrayImageGray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # FFT and centered
    arrayFFTed = np.fft.fft2(arrayImageGray)
    arrayShiftedFFTed = np.fft.fftshift(arrayFFTed)

    # Frequency component power spectrum
    arrayPowerSpectrumFFTRealPart = 20 * np.log(np.absolute(arrayShiftedFFTed))

    arrayBuffer = np.copy(arrayPowerSpectrumFFTRealPart)
    min = np.min(arrayBuffer)
    arrayBuffer[:,:] -= min
    max = np.max(arrayBuffer)
    arrayBuffer[:,:] /= max
    arrayBuffer[:,:] *= 255
    print(arrayBuffer)
    arrayBufferUint8 = arrayBuffer.astype(np.uint8)
    print(arrayBufferUint8.dtype)
    print(arrayBufferUint8.shape)
    cv2.imshow('PowerSpectrum', arrayBufferUint8)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Uncentered and Inverse FFT
    arrayRevertedShiftedFFTed = np.fft.fftshift(arrayShiftedFFTed)
    arrayInvertFFTed = np.fft.ifft2(arrayRevertedShiftedFFTed).real
    arrayBufferUint8 = arrayInvertFFTed.astype(np.uint8)
    print(arrayBufferUint8.dtype)
    print(arrayBufferUint8.shape)

    cv2.imshow('Revered-back Image', arrayBufferUint8)
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

    arrayFaces = cascadeClassifierFace.detectMultiScale(arrayImageGray) # it returns the list of ([tartX, startY, width, height]
    print(arrayFaces)

    colorFace = (255, 0, 0) # color is tuple in CV
    colorEye = (0, 255, 0)
    LineThickness = 2

    for (facePositionX, facePositionY, faceWidth, faceHeight) in arrayFaces:
        cv2.rectangle(arrayImageColor, (facePositionX, facePositionY), (facePositionX + faceWidth, facePositionY + faceHeight), colorFace, LineThickness)
        arrayBufferFace = arrayImageColor[facePositionY: facePositionY + faceHeight, facePositionX: facePositionX + faceWidth]
        arrayBufferFaceGray = arrayImageGray[facePositionY: facePositionY + faceHeight, facePositionX: facePositionX + faceWidth]
        arrayEyes = cascadeClassifierEye.detectMultiScale(arrayBufferFaceGray)
        print(arrayEyes)
        for (eyePositionX, eyePositionY, eyeWidth, eyeHeight) in arrayEyes:
            cv2.rectangle(arrayBufferFace, (eyePositionX, eyePositionY), (eyePositionX + eyeWidth, eyePositionY + eyeHeight), colorEye, LineThickness)

    cv2.imshow('face recogntion', arrayImageColor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # histogram
    arrayImageColor = cv2.imread(filename, cv2.IMREAD_COLOR)
    colors = ('b','g','r') #matlibplot has the color index, https://matplotlib.org/3.1.0/api/colors_api.html#module-matplotlib.colors
    for (indexColor, color) in enumerate(colors): # it provides list of object with index, in this case [(0, 'b'), (1, 'g'), (2, 'r')] 
        # https://docs.opencv.org/master/d1/db7/tutorial_py_histogram_begins.html
        # cv.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
        # images : it is the source image of type uint8 or float32. it should be given in square brackets, ie, "[img]".
        # channels : it is also given in square brackets. It is the index of channel for which we calculate histogram. For example, if input is grayscale image, its value is [0]. For color image, you can pass [0], [1] or [2] to calculate histogram of blue, green or red channel respectively.
        # mask : mask image. To find histogram of full image, it is given as "None". But if you want to find histogram of particular region of image, you have to create a mask image for that and give it as mask. (I will show an example later.)
        # histSize : this represents our BIN count. Need to be given in square brackets. For full scale, we pass [256].
        # ranges : this is our RANGE. Normally, it is [0,256].
        arrayHistogram = cv2.calcHist([arrayImageColor],[indexColor],None,[256],[0,256]) # it retunrs list of channel histogram
        plt.plot(arrayHistogram, color = color)
        plt.xlim([0,256])
    plt.show()



if __name__ == "__main__":
    main()