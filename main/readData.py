import cv2
import matplotlib.pyplot as plt

image = cv2.imread('../data/train/car/0000.jpg')

hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
for i, col in enumerate(['b', 'g', 'r']):
    hist = cv2.calcHist([hsvImage], [i], None, [128], [0, 128])
    plt.plot(hist, color=col)
    plt.xlim([0, 128])

plt.show()

cv2.imshow('Original image', image)
cv2.imshow('HSV image', hsvImage)

cv2.waitKey(0)
cv2.destroyAllWindows()