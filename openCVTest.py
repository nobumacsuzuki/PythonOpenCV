import cv2
import numpy as np

def main():
    filename = "lena_512x512.bmp"
    imageGray = cv2.imread(filename, 0)
    print(imageGray)
    cv2.imshow('B&W image',imageGray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    imageColor = cv2.imread(filename, 1)
    cv2.imshow('Color image',imageColor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

if __name__ == "__main__":
    main()