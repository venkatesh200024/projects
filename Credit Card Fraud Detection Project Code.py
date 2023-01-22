# The dataset for this project was taken from this link: https://www.kaggle.com/mlg-ulb/creditcardfraud

# The topic of credit craud fraud detection drew my interest because in the current day, cashless transactions are mainstream. Cash transactions are secondary to credit/debit card transactions. 
# With this practice, fraudulent activities are also more prevalent. Fraudulent activities are detrimental to banks and customers hurting their bottom line. Every year tens of millions of dollars can be lost due to fraud. 
# Because there are a number of ways to commit credit card fraud, such as stolen credit cards, data breaches, or mail intercepts, catching the small number of fraudulent activities can be tough, but is essential. 
# I decided to use numerous classification algorithms, because we can not assume which fraudulent technique a person may use, so applying algorithms will allow us to better classify the transactions of the dataset.   

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# reading the credit card transaction csv file using pandas
data = pd.read_csv("C:\\Users\\venka\\OneDrive\\Documents\\Downloads//creditcard.csv")
#dropped the Time and Class variables as they were not influential in classification
data.drop('Time', axis = 1, inplace = True)
X = data.drop('Class', axis = 1).values
Y = data['Class'].values
# performed a train test split so we can evaluate the performance of the algorithms
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=42)

# applied the decision tree algorithm to the dataset and computed the accuracy, precision, recall, and f1 score to evaluate the algorithms performance
tree_model = DecisionTreeClassifier(criterion = 'entropy')
tree_model.fit(X_train, Y_train)
tree_yhat = tree_model.predict(X_test)
accuracy_tree = (accuracy_score(Y_test, tree_yhat)) * 100
print("The accuracy of Decision Tree algorithm is " + str(accuracy_tree))
print("Model Precision:", round(precision_score(Y_test, tree_yhat),4))
print("Model Recall:", round(recall_score(Y_test, tree_yhat),4))
print("F1 Score:", round(f1_score(Y_test, tree_yhat),4))
tree_matrix = confusion_matrix(Y_test, tree_yhat, labels = [0, 1]) #printed a confusion matrix to visualize true and false positives as well as true and false negatives
print(tree_matrix)

# applied the KNN algorithm to the dataset and computed the accuracy, precision, recall, and f1 score to evaluate the algorithms performance
knn = KNeighborsClassifier(n_neighbors = 5) #Set the nearest neighbors number as 5
knn.fit(X_train, Y_train)
knn_yhat = knn.predict(X_test)
accuracy_knn = (accuracy_score(Y_test, knn_yhat)) * 100
print("The accuracy of KNN algorithm is " + str(accuracy_knn))
print("Model Precision:", round(precision_score(Y_test, knn_yhat),4))
print("Model Recall:", round(recall_score(Y_test, knn_yhat),4))
print("F1 Score:", round(f1_score(Y_test, knn_yhat),4))
knn_matrix = confusion_matrix(Y_test, knn_yhat, labels = [0, 1])  #printed a confusion matrix to visualize true and false positives as well as true and false negatives
print(knn_matrix)

# applied the logistic regression algorithm to the dataset and computed the accuracy, precision, recall, and f1 score to evaluate the algorithms performance
lr = LogisticRegression()
lr.fit(X_train, Y_train)
lr_yhat = lr.predict(X_test)
accuracy_lr = (accuracy_score(Y_test, lr_yhat)) * 100
print("The accuracy of Logistic Regression algorithm is " + str(accuracy_lr))
print("Model Precision:", round(precision_score(Y_test, lr_yhat),4))
print("Model Recall:", round(recall_score(Y_test, lr_yhat),4))
print("F1 Score:", round(f1_score(Y_test, lr_yhat),4))
lr_matrix = confusion_matrix(Y_test, lr_yhat, labels = [0, 1])  #printed a confusion matrix to visualize true and false positives as well as true and false negatives
print(lr_matrix)

# applied the support vector machine algorithm to the dataset and computed the accuracy, precision, recall, and f1 score to evaluate the algorithms performance
svm = SVC()
svm.fit(X_train, Y_train)
svm_yhat = svm.predict(X_test)
accuracy_svm= (accuracy_score(Y_test, svm_yhat)) * 100
print("The accuracy of SVM algorithm is" + str(accuracy_svm))
print("Model Precision:", round(precision_score(Y_test, svm_yhat),4))
print("Model Recall:", round(recall_score(Y_test, svm_yhat),4))
print("F1 Score:", round(f1_score(Y_test, svm_yhat),4))
svm_matrix = confusion_matrix(Y_test, svm_yhat, labels = [0, 1]) #printed a confusion matrix to visualize true and false positives as well as true and false negatives
print(svm_matrix)










