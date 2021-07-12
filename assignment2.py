from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
import numpy as np
import pandas as pd

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

classifiers = {
    'linear SVC': SVC(kernel='linear', probability=True,random_state=0, max_iter = 500),
    'poly SVC': SVC(kernel='poly', probability=True,random_state=0, max_iter = 500)
}

n_classifiers = len(classifiers)

for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(train_data, train_labels)
    pred = classifier.predict(test_data)
    accuracy = accuracy_score(test_labels, pred)
    print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
    print(classification_report(test_labels, pred, target_names=classes))
