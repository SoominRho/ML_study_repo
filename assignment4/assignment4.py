from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

train_data = np.fromfile('train-images-idx3-ubyte', dtype = np.uint8)
train_data = np.delete(train_data, range(16))
train_data = train_data.reshape(-1, 784)

train_labels = np.fromfile('train-labels-idx1-ubyte', dtype = np.uint8)
train_labels = np.delete(train_labels, range(8))
train_labels = train_labels.reshape(-1, 1)

test_data = np.fromfile('t10k-images-idx3-ubyte', dtype = np.uint8)
test_data = np.delete(test_data, range(16))
test_data = test_data.reshape(-1, 784)

test_labels = np.fromfile('t10k-labels-idx1-ubyte', dtype = np.uint8)
test_labels = np.delete(test_labels, range(8))
test_labels = test_labels.reshape(-1, 1)

total_data = np.concatenate((train_data, test_data), axis=0)

pca = PCA(n_components = 2)
total_2D = pca.fit_transform(total_data)
train_2D = total_2D[0:60000]
test_2D = total_2D[60000:70000]
print(train_2D.shape)
print(test_2D.shape)
train_with_label = np.concatenate([train_2D, train_labels], axis=1)
test_with_label = np.concatenate([test_2D, test_labels], axis=1)
train_labels = train_labels.reshape(-1)
test_labels = test_labels.reshape(-1)
train_with_label = pd.DataFrame(data = train_with_label , columns = ['x', 'y', 'labels'])
test_with_label = pd.DataFrame(data = test_with_label , columns = ['x', 'y', 'labels'])
train_with_label = train_with_label.astype({'labels': 'object'})
test_with_label = test_with_label.astype({'labels': 'object'})


#PCA 대신 TSNE를 이용
"""
tsne = TSNE(n_components = 2)
total_2D = tsne.fit_transform(total_data)
train_2D = total_2D[0:60000]
test_2D = total_2D[60000:70000]
print(train_2D.shape)
print(test_2D.shape)
train_with_label = np.concatenate([train_2D, train_labels], axis=1)
test_with_label = np.concatenate([test_2D, test_labels], axis=1)
train_labels = train_labels.reshape(-1)
test_labels = test_labels.reshape(-1)
train_with_label = pd.DataFrame(data = train_with_label , columns = ['x', 'y', 'labels'])
test_with_label = pd.DataFrame(data = test_with_label , columns = ['x', 'y', 'labels'])
train_with_label = train_with_label.astype({'labels': 'object'})
test_with_label = test_with_label.astype({'labels': 'object'})
"""

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
label_decodes = {
    0 : 'T-shirt/top',
    1 : 'Trouser',
    2 : 'Pullover',
    3 : 'Dress',
    4 : 'Coat',
    5 : 'Sandal',
    6 : 'Shirt',
    7 : 'Sneaker',
    8 : 'Bag',
    9 : 'Ankle boot'
}

for i in range(60000):
    train_with_label.at[i,'labels'] = label_decodes[train_labels[i]]
print(train_with_label.head())

for i in range(10000):
    test_with_label.at[i,'labels'] = label_decodes[test_labels[i]]
print(test_with_label.head())


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,2,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA for Train data', fontsize = 20)
ax.set(xlim=[-3000., 3000.], ylim=[-3000., 3000.])

for label in range(10):
    indice = train_with_label['labels'] == label_decodes[label]
    ax.scatter(train_with_label.loc[indice, 'x'], train_with_label.loc[indice, 'y'], s = 30)
ax.legend(labels)
ax.grid()

"""
ax2 = fig.add_subplot(1,2,2) 
ax2.set_xlabel('Principal Component 1', fontsize = 15)
ax2.set_ylabel('Principal Component 2', fontsize = 15)
ax2.set_title('2 Component PCA for Test data', fontsize = 20)
ax2.set(xlim=[-3000., 3000.], ylim=[-3000., 3000.])

for label in range(10):
    indice = test_with_label['labels'] == label_decodes[label]
    ax2.scatter(test_with_label.loc[indice, 'x'], test_with_label.loc[indice, 'y'], s = 30)
ax2.legend(labels)
ax2.grid()
plt.show()
"""

# k = 10으로 클러스터링
train_2D = pd.DataFrame(train_2D, columns = ['x', 'y'])
centroids = train_2D.sample(10, random_state=1)

distance = sp.spatial.distance.cdist(train_2D, centroids, "euclidean")
cluster_num = np.argmin(distance, axis=1)
result = train_2D.copy()
result["cluster"] = np.array(cluster_num)
print(result.head())

for i in range(10):
    centroids = result.groupby("cluster").mean()
    distance = sp.spatial.distance.cdist(train_2D, centroids, "euclidean")
    cluster_num = np.argmin(distance, axis=1)
    result = train_2D.copy()
    result["cluster"] = np.array(cluster_num)


ax2 = fig.add_subplot(1,2,2) 
ax2.set_xlabel('Principal Component 1', fontsize = 15)
ax2.set_ylabel('Principal Component 2', fontsize = 15)
ax2.set_title('Clustering results', fontsize = 20)
ax2.set(xlim=[-3000., 3000.], ylim=[-3000., 3000.])

for label in range(10):
    indice = result['cluster'] == label
    ax2.scatter(result.loc[indice, 'x'], result.loc[indice, 'y'], s = 30)
ax2.scatter(centroids['x'], centroids['y'], c = 'black')
ax2.grid()
plt.show()
