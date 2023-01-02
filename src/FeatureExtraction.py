import os
import time

# Read an image and convert it to the HSV color space, using OpenCV .
import cv2
import numpy as np
# Import label encoder
from sklearn import preprocessing

# Image data constants
from sklearn.preprocessing import StandardScaler

DIMENSION = 32
DIMENSIONS = (DIMENSION, DIMENSION)
# ROOT_DIR = "../data/test/"
ROOT_DIR2 = "../../project1data/"
DEER = "deer"
FROG = "frog"
AIRPLANE = "airplane"
AUTOMOBILE = "automobile"
BIRD = "bird"
HORSE = "horse"
TRUCK = "truck"
CLASSES = [DEER, FROG, AIRPLANE, AUTOMOBILE, BIRD, HORSE, TRUCK]
DATASETTYPE = ["test", "train"]

def covertImageToFlattenHSV(dirName, fileName):
    # https://techtutorialsx.com/2019/11/08/python-opencv-converting-image-to-hsv/
    # Read the image into the memory
    image = cv2.imread(dirName + fileName)
    # https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
    # Scaling the image using OpenCV binary extension loader to 32 X 32 pixel with cv2.INTER\_LINEAR interpolation method which is the default OpenCV method.
    image = cv2.resize(image, DIMENSIONS, interpolation=cv2.INTER_LINEAR)
    # Convert the image from RGB color space to HSV color space.
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # converting the HSV m x n matrix into a array of values.
    # flattenImage = hsvImage.flatten()

    # cv2.imshow('Original image',image)
    # cv2.imshow('HSV image', hsvImage)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return hsvImage


def generateImagesMatrix(dsType):
    Xdata = []
    Ydata = []
    histogramData=[]
    #Iterate over the included training and testing dataset images and create the image in the memory.
    for clazz in CLASSES:
        for fileName in os.listdir(ROOT_DIR2 + dsType + "/" + clazz + "/"):
            histogramData=[]
            flattenImage = covertImageToFlattenHSV(ROOT_DIR2 + dsType + "/"+ clazz + "/", fileName)
            h1Blue = cv2.calcHist([flattenImage], [0], None, [16], [0, 256]).ravel()
            h2Green = cv2.calcHist([flattenImage], [1], None, [16], [0, 256]).ravel()
            h3Red = cv2.calcHist([flattenImage], [2], None, [16], [0, 256]).ravel()
            # Flatten using extending list by appending elements for each channel
            histogramData.extend(h1Blue)
            histogramData.extend(h2Green)
            histogramData.extend(h3Red)

            Xdata.append(histogramData)
            Ydata.append(clazz)

    Xdata = np.array(Xdata)
    # print(Xdata)
    # print("---------------------")
    Ydata = np.array(Ydata)
    # print(Ydata)
    # https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python/
    # label_encoder object knows how to understand word labels.
    label_encoder = preprocessing.LabelEncoder()
    # Encode labels in column 'species'.
    Ydata = label_encoder.fit_transform(Ydata)
    return Xdata, Ydata

#       Ydata --> Labels
#       Xdata --> Images data
def generateFeaturesFiles(dataSetType, Xdata, Ydata):
    line = ""
    counter1 = 0
    counter2 = 1
    time1 = int(time.time())
    print(f"The current time in milliseconds: {time1}")
    df = open(ROOT_DIR2 + dataSetType + str(time1), 'w')
    for idx in Xdata:
        df.write(str(Ydata[counter1]))
        # print ("str(Ydata[idx]) " + str(Ydata[idx]))
        for idx2 in idx:
            df.write(' ' + str(counter2) + ':' + str(idx2))
            counter2 = counter2 + 1
        df.write('\n')
        counter1 = counter1 + 1
        counter2 = 1
        line = ""
    print(f"The current time in milliseconds: {int(time.time()) - time1}")

if __name__ == "__main__":
    #Iterate over the included training and testing dataset images and create the image in the memory.
    for dsType in DATASETTYPE:
        (Xdata, Ydata) = generateImagesMatrix(dsType)
        # Normalization/scaling
        Xdata=StandardScaler().fit_transform(Xdata)
        generateFeaturesFiles(dsType, Xdata, Ydata)
