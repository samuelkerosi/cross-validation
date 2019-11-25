import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score, cross_validate

trainingFile = np.genfromtxt("irisTraining.txt")
X_train = trainingFile[:, -1]
y_train = trainingFile[:, -1]

fit = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
# print(fit)

cross_validation_score = cross_val_score(fit, X_train, y_train, cv=10)
print(cross_validation_score)

cross_validation_mean_score = np.mean(cross_validation_score)
print(cross_validation_mean_score)

# output file
outfile = open("crossvalidatio.txt", "w")
outfile.write("Cross Validation Score \n")
outfile.write(" \n")
outfile.write(str(cross_validation_score))
outfile.write(" \n")
outfile.write("Cross Validation Mean Score \n")
outfile.write(str(cross_validation_mean_score))
