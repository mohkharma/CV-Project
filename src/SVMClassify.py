# Read an image and convert it to the HSV color space, using OpenCV .
import libsvm.commonutil
import libsvm.svm
import libsvm.svmutil
from libsvm import svmutil, svm
# Import label encoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# : Linear, Polynomial of 2nd degree, and Radial Basis Function (RBF)

LINEAR = 0
POLY = 1
RBF = 2
SIGMOID = 3
PRECOMPUTED = 4
CHISQUAREDNORM = 5
ROOT_DIR = "../../project1data/"
TRAIN_FILENAME = "train1672428003"
TEST_FILENAME = "test1672427916"
DEER = "deer"
FROG = "frog"
AIRPLANE = "airplane"
AUTOMOBILE = "automobile"
BIRD = "bird"
HORSE = "horse"
TRUCK = "truck"
CLASSES = [DEER, FROG, AIRPLANE, AUTOMOBILE, BIRD, HORSE, TRUCK]

def SVMClassify(kernel):
    # prepare libsvm parameters based on the selected kernal
    param = libsvm.svm.svm_parameter("-q")
    param.probability = 1
    if (kernel == LINEAR):
        param.kernel_type = LINEAR
        param.C = .01
    if (kernel == POLY):
        param.kernel_type = POLY
        param.C = .01
        param.gamma = .00000001
    if (kernel == RBF):
        param.kernel_type = RBF
        param.C = .01
        param.gamma = .00000001
    if (kernel == SIGMOID):
        param.kernel_type = SIGMOID
        param.C = .01
        param.gamma = .00000001
    if (kernel == PRECOMPUTED):
        param.kernel_type = PRECOMPUTED
        param.C = .01
        param.gamma = .00000001
    if (kernel == CHISQUAREDNORM):
        param.kernel_type = CHISQUAREDNORM
        param.C = .01
        param.gamma = .00000001

    # svm problem read for training data
    y_train, x_train = svmutil.svm_read_problem(ROOT_DIR + TRAIN_FILENAME)
    # svm problem read for testing data
    y_test, x_test = svmutil.svm_read_problem(ROOT_DIR + TEST_FILENAME)

    # Create svm problem based on the training data to pass it to the training model
    prob = svm.svm_problem(y_train, x_train)
    model = svmutil.svm_train(prob, param)

    # Test prediction
    #    The return tuple contains
    #    p_labels: a list of predicted labels
    #    p_acc: a tuple including  accuracy (for classification), mean-squared
    #           error, and squared correlation coefficient (for regression).
    #    p_vals: a list of decision values or probability estimates (if '-b 1'
    #            is specified). If k is the number of classes, for decision values,
    #            each element includes results of predicting k(k-1)/2 binary-class
    #            SVMs. For probabilities, each element contains k values indicating
    #            the probability that the testing instance is in each class.
    #            Note that the order of classes here is the same as 'model.label'
    #            field in the model structure.
    p_labels, p_acc, p_vals = svmutil.svm_predict(y_test, x_test, model, "-q -b 1")

    print("-----------------------------Model accuracy -------------------------------------")
    print(p_acc[0])
    f1Score = f1_score(y_test, p_labels, average='micro')
    print("-----------------------------F1 score-------------------------------------")
    print(f1Score)

    # Generate the confugion matrix
    confusionMatrix = confusion_matrix(y_test, p_labels)

    print("Confusion matrix =>", confusionMatrix)


if __name__ == "__main__":
    SVMClassify(CHISQUAREDNORM)
