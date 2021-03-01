import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_decision_regions



bankdata = pd.read_csv(r"D:\Doctora99\Doctora-term1-9908\ML.DrManthouri\HW\SVM\bill_authentication.csv")
bankdata.shape
bankdata.head()

X = bankdata.drop('Class', axis=1)
y = bankdata['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

###############################
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print("############ SVC(kernel='linear') ############")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

###############################
svclassifier_poly = SVC(kernel='poly')
svclassifier_poly.fit(X_train, y_train)

y_pred = svclassifier_poly.predict(X_test)

print("############ SVC(kernel='poly') ############")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#++++++++++++++++++++++++++++++++++++++++++++++
svclassifier_rbf = SVC(kernel='rbf')
svclassifier_rbf.fit(X_train, y_train)

y_pred = svclassifier_rbf.predict(X_test)

print("############ SVC(kernel='rbf') ############")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#***********************************************
svclassifier_sigmoid = SVC(kernel='sigmoid')
svclassifier_sigmoid.fit(X_train, y_train)

y_pred = svclassifier_sigmoid.predict(X_test)

print("############ SVC(kernel='sigmoid') ############")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

###############################################
######################################
###@@@@@@ WithOut PCA ***@@@@@@@

svclassifier = SVC(C=100,gamma=0.0001)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print("############ SVC(C=100,gamma=0.0001) ############")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

################################################################################
#######################################################################
######## Using PCA ************** Plot decision boundary #########

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
#explained_variance = pca.explained_variance_ratio_

X_train2 = X_train
y_train2 = y_train
y_train2 = y_train2.as_matrix()
X_train2 = pca.fit_transform(X_train2)

svclassifier = SVC(C=100,gamma=0.0001)
svclassifier.fit(X_train2, y_train2)
plot_decision_regions(X_train2, y_train2, clf=svclassifier, legend=2)

plt.xlabel(X.columns[0], size=14)
plt.ylabel(X.columns[1], size=14)
plt.title('SVM Decision Region Boundary', size=16)


y_test2 = y_test
y_test2 = y_test2.as_matrix()
X_test2 = X_test
X_test2 = pca.fit_transform(X_test2)
y_pred2 = svclassifier.predict(X_test2)
print("############ SVC(C=100,gamma=0.0001) wtit PCA ############")
print(confusion_matrix(y_test2,y_pred2))
print(classification_report(y_test2,y_pred2))

################ Get support vectors ################ 
support_vectors = svclassifier.support_vectors_
plt.figure()

# Visualize support vectors
plt.scatter(X_train2[:,0], X_train2[:,1])
plt.scatter(support_vectors[:,0], support_vectors[:,1], color='red')
plt.title('Linearly separable data with support vectors')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
