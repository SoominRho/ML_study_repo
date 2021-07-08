import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

raw_train = np.load('regression.npy')
raw_test = np.load('regression_test.npy')

x_train = raw_train[:,0:1]
y_train = raw_train[:,1:2]
x_test = raw_test[:,0:1]
y_test = raw_test[:,1:2]
"""
plt.scatter(x_train, y_train) 
plt.xlabel('input')
plt.ylabel('output')
plt.show()

plt.scatter(x_test, y_test) 
plt.xlabel('input')
plt.ylabel('output')
plt.show()
"""

plt.subplot(2,1,1)
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(x_train, y_train)
predict = pipeline.predict(x_test)
df = pd.DataFrame({'x': x_test[:,0], 'y': predict[:,0]})
df.sort_values(by='x',inplace = True)
points = pd.DataFrame(df).to_numpy()
plt.plot(points[:, 0], points[:, 1],color="blue", linewidth=3)
plt.xlabel('input')
plt.ylabel('output')
plt.scatter(x_train,y_train, color="black")
print(math.sqrt(mean_squared_error(y_test, predict)))

plt.subplot(2,1,2)
pipeline = make_pipeline(PolynomialFeatures(6), LinearRegression())
pipeline.fit(x_train, y_train)
predict = pipeline.predict(x_test)
df = pd.DataFrame({'x': x_test[:,0], 'y': predict[:,0]})
df.sort_values(by='x',inplace = True)
points = pd.DataFrame(df).to_numpy()
plt.plot(points[:, 0], points[:, 1],color="blue", linewidth=3)
plt.xlabel('input')
plt.ylabel('output')
plt.scatter(x_train,y_train, color="black")
print(math.sqrt(mean_squared_error(y_test, predict)))
plt.show()


