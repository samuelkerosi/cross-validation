import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier
trainingFile = np.genfromtxt("irisTraining.txt")
X_train = trainingFile[:, -1]
y_train = trainingFile[:, -1]

knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)
cross_validation = cross_validate(knn_classifier, X_train, y_train, cv=10)

# output file
outfile = open("crossvalidation.txt", "w")
outfile.write("Cross Validation \n")
outfile.write(" \n")
outfile.close()
