Support Vector Machines (SVMs) are a sort of supervised machine learning al-
gorithm that can be employed for image classification tasks. SVMs function by
finding the hyperplane in a high-dimensional space that maximally separates var-
ious categories. In this report, we use a dataset called CIFAR-10, it is a widely
used dataset in image classification problems and is composed of 60,000 colored
32x32 images. To use an SVM for image classification, Initially, we preprocess
CIFAR-10 dataset images and extract features that are relevant to the classifi-
cation task. We use a color histogram approach in this report. After extracting
the relevant features from the images, we train an SVM classifier by feeding it a
dataset of labeled images and their corresponding features. The SVM uses this
dataset to learn the decision boundary that separates the different classes. After
the SVM has been trained, now we can then use it to classify new images by
extracting features from the images and then using the trained SVM to predict
the class of the image based on which side of the decision boundary it falls on.
There are number of parameters that can be tuned when training an SVM,
such as the type of kernel to use and the penalty parameter C. It is generally a
good idea to use cross-validation to find the optimal values for these parameters.
Overall, using SVMs for image classification can be a powerful and effective
approach, but it is important to carefully preprocess and select relevant features
for the classification task. We have extended the supported kernels to chi-squared
(2) beside the existing kernels in the open-source LIBSVM project.



**2 Methodology**

2.1 Feature extraction
Feature extraction is the process of extracting relevant information from an
image or other data source and representing it in a compact form that can
be used for tasks such as object recognition, classification, or analysis. Feature
extraction techniques aim to extract features that are discriminative, meaning
that they can effectively distinguish between different classes or categories. These
features can be based on various image characteristics, such as edges, corners,
texture, or color. Images should be represented numerically so that learners can
construct decision boundaries in order to acquire certain class labels for the task
of classifying images. As much of the information from the images should be
retained via these characteristics.
In this report, we represent the images using color histograms. Therefore,
each image will have its RGB (Red, Green, and Blue) histogram extracted in
order to be represented. Additionally, we have created the HSV (Hue, Saturation,
Luminance), which will serve as the ultimate illustration of each image’s color
histogram. In the following, we provide the process that we have applied for the
feature extraction phase:

– Download the dataset from https://github.com/YoongiKim/CIFAR-10-images.

– Select 7 classes randomly that will be included into our classification task.
Selected classes are DEER, FROG, AIRPLANE, AUTOMOBILE, BIRD,
HORSE, and TRUCK.

– Write python code to:

• Iterate over the included training and testing dataset images and create
the image in the memory.

• Scaling the image using OpenCV binary extension loader to 32 X 32
pixel with cv2.INTER LINEAR interpolation method which is the de-
fault OpenCV method.

• Convert the image from RGB color space to HSV color space. One reason
that HSV is often used is that it separates the color information (hue)
from the intensity information (saturation and value), which makes it
easier to adjust the colors in an image without changing the overall
brightness or intensity. It is also easier for humans to think about colors
in terms of hue, saturation, and value, which makes it a useful represen-
tation for user interfaces and other applications where color selection is
important.

• Flattening the HSV matrix and collapsing it into one dimension array.

• Perform label encoding in order to give each class a unique number using
sklearn.preprocessing‘ module.

• Store the flattened training and testing images data into separated text
files.

    – Verify the generated file’s validity by running checkdata.py python validator
provided by the LIBSVM project.


**2.2 Grid search for hyper-parameters tuning**
Grid search is a technique for tuning a model’s hyper-parameters. In order to do
this, a grid of hyper-parameter values must be specified. A model then needs to
be trained and tested for each combination of these values. Consider the situation
where we are trying to determine the ideal values for the hyper-parameters C
and gamma while training a support vector machine (SVM) with a Radial Basis
Function(RBF) kernel. A hyper-parameter called gamma controls the model’s
kernel function’s width. Gamma in the aforementioned example controls the
degree to which each training example has an impact on the model. The model
will attempt to fit the training data more closely if gamma has a larger value,
which suggests that each training example has a stronger influence. If the model
becomes overly sensitive to the noise in the training data, this can potentially
result in over-fitting. The regularization hyper-parameter C regulates the cost
of wrongly classifying training examples. The following sample grid search runs
have been used to find the best hyper-parameters against different kernels.

    
    # Linear
    >python ./grid.py -log2c -2,1,1 -log2g -1,0,1 -v 5 -m 1000
    -t 0 -png test1672509709PNG1 "~\project1data\train1672509709"
    #Output
    -1.0 0.0 36.1571 (best c=0.5, g=1.0, rate=36.1571)
    # Linear with the use of the scaled dataset
    >python ./grid.py -t 0 "~\project1data\train1672509709"
    -2.0 0.0 36.6029 (best c=0.25, g=1.0, rate=36.6029)
    # POLY
    >python ./grid.py -log2c -2,2,1 -log2g -1,0,1 -v 5 -m 1000
    -t 1 -png test1672509709PNG1 "~\project1data\train1672509709"
    #Output
    1.0 -1.0 48.0571 (best c=0.25, g=1.0, rate=48.0571)
    # POLY with the use of the scaled dataset
    >python ./grid.py -t 1 "~\project1data\train1672509709"
    #Output
    -1 -7 34.36 (best c=32.0, g=0.0078125, rate=43.0429)
    # RBF
    >python ./grid.py -log2c -2,1,1 -log2g -1,0,1 -v 5 -m 1000
    -t 2 -png test1672509709PNG2 \"~\project1data\train1672509709"
    #Output
    -2.0 -1.0 14.2857 (best c=0.25, g=1.0, rate=14.2857)
    # RBF with the use of the scaled dataset
    >python ./grid.py -t 2 \"~\project1data\train1672509709"
    #Output
    -1 -1 19.8686 (best c=32.0, g=0.0078125, rate=46.7571)
    # Chi-squared run while there is no C or gamma included
    
    #into the kernel function implementation:
    >python ./grid.py -log2c -2,2,1 -log2g -1,0,1 -v 5 -m 1000
    -t 5 -png train1672509807PNG5 "~\project1data\train1672509709"
    #Output
    -1.0 0.0 36.1571 (best c=0.5, g=1.0, rate=36.1571)
    # Chi-squared with the use of the scaled dataset
    >python ./grid.py -t 5 "~\project1data\train1672509709"
    #Output
    9 1 9.25143 (best c=0.5, g=0.0078125, rate=9.30286)


**2.3 Classification model and results**

The existing supported kernels by LIBSVM are Linear, Polynomial of 2nd degree,
and Radial Basis Function (RBF). In order to add the Chi-squared (2) kernel,
we have followed the following steps:

– Download the LIBSVM source code from https://www.csie.ntu.edu.tw/ cjlin/
libsvm/download.

– Modified svm.h and svm.cpp files by adding CHISQUAREDNORM as a new
kernel type with implementing the Chi-squared (2) kernel functions. Modi-
fied version can be found on https://github.com/mohkharma/CV-Project/tree/master/src/libsvm

– Run KNN-based classifier to use it as a base for the comparison of the results.
https://github.com/mohkharma/kNN_CiFAR10dataset
/blob/master/knn_CIFAR10_V2.ipynb

– The accuracy of KNN is as follows in table 1
    
    Value of K Accuracy F1-Score
    5 32.7143 32.7143
    7 33.1143 33.1143
    10 33.2571 33.2571
    15 33.4286 33.4286
    Table 1. KNN runs using different K values
In order to generate SVM training model so we can use it later in the testing,
we use the following for each kernel:

    >.\svm-train.exe -t 1 -g 1 -c 0.25 train1672428003
    >.\svm-train.exe -t 0 -g 0.0078125 -c 32 train1672428003
    >.\svm-train.exe -t 2 -g 32 -c 0.0078125 train1672428003
    >.\svm-train.exe -t 2 -g 0.5 -c 0.0078125 train1672428003


The accuracy, confusion matrix, and F1-score for each generated model from
the above on the test dataset are as follows:

– Linear:

    Model accuracy => 37.51428571428572
    F1 score => 37.514285714285717
    Confusion matrix => [[669 70 50 46 43 45 77]
    [179 406 53 37 87 55 183]
    [191 62 96 259 255 80 57]
    [ 92 61 71 347 319 74 36]
    [ 50 41 61 179 535 103 31]
    [ 79 82 88 179 245 218 109]
    [186 238 60 26 74 61 355]]
When we trained the model using the same dataset but before scaling
and using the flattened HSV values, the accuracy was 34.35714285714286
and F1-score was 34.35714285714286.

– POLY:

    Model accuracy => 34.385714285714286
    F1 score => 34.385714285714286
    Confusion matrix => [[422 7 19 10 9 177 356]
    [ 26 44 1 1 18 211 699]
    [100 9 78 106 82 423 202]
    [ 35 5 30 194 124 468 144]
    [ 7 5 13 74 239 531 131]
    [ 12 8 2 19 27 660 272]
    [ 25 11 1 2 6 185 770]]
But when we trained the model using the same dataset but before scaling
and using the flattened HSV values, the accuracy was 46.42857142857143
and F1-score was 46.42857142857143

– Radial Basis Function (RBF):

    Model accuracy => 48.31428571428572
    F1 score => 48.314285714285715
    Confusion matrix => [[620 67 96 38 28 71 80]
    [ 68 526 20 16 43 94 233]
    [156 54 280 180 178 115 37]
    [ 81 48 121 368 228 132 22]
    [ 24 38 80 141 581 110 26]
    [ 62 82 64 82 109 491 110]
    [ 78 242 17 19 34 94 516]]
But when we trained the model using the same dataset but before scaling
and using the flattened HSV values, the accuracy was 14.285714285714285
and F1-score was 14.285714285714285.

– CHI2:

    Model accuracy => 14.428571428571429
    F1 score => 0.1442857142857143
    Confusion matrix => [[338 134 20 107 150 54 197]
    [484 121 22 110 93 42 128]
    [376 96 24 182 150 54 118]
    [342 94 30 212 162 59 101]
    [353 119 24 222 139 58 85]
    [438 72 29 167 126 54 114]
    [496 114 23 109 96 40 122]]
But when we trained the model using the same dataset but before scaling
and using the flattened HSV values, the accuracy was 45.785714285714285
and F1-score was 45.785714285714285


**2.4 Discussion and Conclusion**
Based on the above accuracy, F1 score, and confusion matrices. The best kernel
accuracy performance was found to be the Radial Basis Function(RBF) kernel.
The obtained accuracy and F1-Score are equal to 48.31 since the dataset is a
balanced dataset. The worse results were obtained using a CHI-Squared kernel
where the accuracy and F1-Score are equal to 14.43. Also, when we compared the
above results with another conducted experiment’s results that used the flattened
HSV data without scaling, we noticed better accuracy achieved by CHI-Squared
and Polynomial kernels. We believe the better performance of CHI-Squared and Polynomial kernels is due to a large number of features compared to the number
of features generated from calculating the image histograms. Compared with
KNN classifier, KNN best accuracy reached 33.4286 which comes in the third-
best classifier performance after the RBF and Linear kernels.