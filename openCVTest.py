import cv2

def main():
    filename = "lena_512x512.bmp"
    gry = cv2.imread(filename, 0)
    print(gry)
    cv2.imshow('image',gry)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    color = cv2.imread(filename, 1)
    cv2.imshow('image',color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()