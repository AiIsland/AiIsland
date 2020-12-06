# Support Vector Regression (SVR)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('morocco_covid19-data.csv')
X = np.arange(270)    # creating x array refers to 270 days from first covid19 case till 2020-11-26
X = np.reshape(X, (-1, 1))
y =dataset.iloc[1:, 2].values   # take the number of cases from the dataset
y=np.flip(y)


y = np.reshape(y, (-1, 1))


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()     # scale the training values with Standardization technique 
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result
print("-----------svm prediction for 2020-11-27------")
print(sc_y.inverse_transform(regressor.predict(sc_X.transform([[271]])))) # predict to 271 day wich refer to 2020-11-27

# Visualising the SVR results
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')

plt.title('daily cases of covid 19 in morroco')
plt.xlabel('number of days since first case')
plt.ylabel('contamination number')
plt.show()
