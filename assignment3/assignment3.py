from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import  MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--classifier', type=str )
args  = parser.parse_args()

train_data = np.fromfile('train-images-idx3-ubyte', dtype = np.uint8)
train_data = np.delete(train_data, range(16))
train_data = train_data.reshape(-1, 784)
print(train_data.shape)

train_labels = np.fromfile('train-labels-idx1-ubyte', dtype = np.uint8)
train_labels = np.delete(train_labels, range(8))
print(train_labels.shape)

test_data = np.fromfile('t10k-images-idx3-ubyte', dtype = np.uint8)
test_data = np.delete(test_data, range(16))
test_data = test_data.reshape(-1, 784)
print(test_data.shape)

test_labels = np.fromfile('t10k-labels-idx1-ubyte', dtype = np.uint8)
test_labels = np.delete(test_labels, range(8))
print(test_labels.shape)

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ConvergenceWarning: Solver terminated early (max_iter=500).  Consider pre-processing your data with StandardScaler warnings.warn('Solver terminated early (max_iter=%i).'
# iteration을 키워야하는듯?
  
classifiers = {
    'rbf_SVC': SVC(kernel='rbf', probability=True,random_state=0, max_iter = 10000),        # 88.3%
    'poly_SVC': SVC(kernel='poly', probability=True,random_state=0, max_iter = 10000),      # 85.5%
    'KNN_classifier_5': KNeighborsClassifier(n_neighbors = 5),                              # 85.5%
    'KNN_classifier_10': KNeighborsClassifier(n_neighbors = 10),                            # 85.2%
    'KNN_classifier_15': KNeighborsClassifier(n_neighbors = 15),                            # 84.6%
    'Multinomial_Naive_Bayes': MultinomialNB(),                                             # 65.5%
    'Random_Forest_50': RandomForestClassifier(n_estimators=50),                            # 87.2%
    'Random_Forest_100': RandomForestClassifier(n_estimators=100)                           # 87.6%
}

print(args.classifier)
print(classifiers[args.classifier])
classifiers[args.classifier].fit(train_data, train_labels)
pred = classifiers[args.classifier].predict(test_data)
accuracy = accuracy_score(test_labels, pred)
print("Accuracy (train) for %s: %0.1f%% " % (args.classifier, accuracy * 100))
print(classification_report(test_labels, pred, target_names=classes))
