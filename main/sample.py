# Import the necessary libraries
from sklearn import datasets
from sklearn import svm

# Load the CIFAR-10 dataset
cifar10 = datasets.load_cifar10()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(cifar10.data, cifar10.target, test_size=0.2)

# Create a support vector machine (SVM) classifier
clf = svm.SVC(gamma='scale')

# Train the classifier using the training data
clf.fit(X_train, y_train)

# Test the classifier using the test data
accuracy = clf.score(X_test, y_test)

print("Accuracy:", accuracy)