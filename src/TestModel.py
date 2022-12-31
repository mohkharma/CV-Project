# Generate the training model by:
# libsvm\windows> .\svm-train.exe -t 1 C:\M.kharma_data\PhD\03-Semester-2022\COMP9318-ML\bk\test1672427916
from python.libsvm import svmutil, svm
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
model = svmutil.svm_load_model("../../project1data/test1672427916.model")

y_test, x_test = svmutil.svm_read_problem("C:/M.kharma_data/PhD/03-Semester-2022/COMP9318-ML/bk/test1672427916")

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