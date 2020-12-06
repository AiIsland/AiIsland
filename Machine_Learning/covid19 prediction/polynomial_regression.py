# Polynomial and linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('morocco_covid19-data.csv')
X = np.arange(270)   # creating x array refers to 270 days from first covid19 case till 2020-11-26
X = np.reshape(X, (-1, 1))
y =dataset.iloc[1:, 2].values   # take the number of cases from the dataset
y=np.flip(y)
y = np.reshape(y, (-1, 1))


# Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')

plt.title('daily cases of covid 19 in morroco')
plt.xlabel('number of days since first case')
plt.ylabel('contamination number')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('daily cases of covid 19 in morroco')
plt.xlabel('number of days since first case')
plt.ylabel('contamination number')
plt.show()


# Predicting a new result with Linear Regression

print("-----------Linear Regression prediction for 2020-11-27------")
print(lin_reg.predict([[271]]))

# Predicting a new result with Polynomial Regression
print("-----------Polynomial Regression prediction for 2020-11-27------")
print(lin_reg_2.predict(poly_reg.fit_transform([[271]])))