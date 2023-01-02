# Generate the training model by:
# libsvm/windows> ./svm-train.exe -t 1 C:/M.kharma_data/PhD/03-Semester-2022/COMP9318-ML/bk/test1672427916
import os
import sys
import random
from libsvm.python.libsvm import svmutil, svm
# Import label encoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report

DEER = "deer"
FROG = "frog"
AIRPLANE = "airplane"
AUTOMOBILE = "automobile"
BIRD = "bird"
HORSE = "horse"
TRUCK = "truck"
CLASSES = [DEER, FROG, AIRPLANE, AUTOMOBILE, BIRD, HORSE, TRUCK]
model = svmutil.svm_load_model("../../project1data/train1672696618_1.model")

rand_list=[]
y_test2 = []
x_test2 = []
for i in range(0,200):
    n=random.randint(0,4999)
    rand_list.append(n)

y_test, x_test = svmutil.svm_read_problem("C:/M.kharma_data/PhD/03-Semester-2022/COMP9318-ML/project1data/test1672696575")
print ("xxxxxxxxxxx" )

if False:
    for i in rand_list:
        x_test2.append(x_test[i])
        y_test2.append(y_test[i])
        x_test2.append(x_test[i + (5000 * 1)] )
        y_test2.append(y_test[i+ (5000 * 1)] )
        x_test2.append(x_test[i + (5000 * 2)] )
        y_test2.append(y_test[i+ (5000 * 2) ])
        x_test2.append(x_test[i + (5000 * 3)] )
        y_test2.append(y_test[i+ (5000 * 3)] )
        x_test2.append(x_test[i + (5000 * 4)] )
        y_test2.append(y_test[i+ (5000 * 4)] )
        x_test2.append(x_test[i + (5000 * 5)] )
        y_test2.append(y_test[i+ (5000 * 5)] )
        x_test2.append(x_test[i + (5000 * 6)] )
        y_test2.append(y_test[i+ (5000 * 6)] )

    x_test = x_test2
    y_test = y_test2
print ("xxxxxxxxxxx")


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
p_labels, p_acc, p_vals = svmutil.svm_predict(y_test, x_test, model, "-q -b 0")


print("-----------------------------Model accuracy -------------------------------------")
print(p_acc[0])
f1Score = f1_score(y_test, p_labels, average='micro')
print("-----------------------------F1 score-------------------------------------")
print(f1Score)

# Generate the confugion matrix
confusionMatrix = confusion_matrix(y_test, p_labels)

print("Confusion matrix =>", confusionMatrix)