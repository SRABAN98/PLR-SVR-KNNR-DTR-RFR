#Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Import the dataset
dataset = pd.read_csv(r"C:\Users\dell\OneDrive\Documents\Data Science\1.SVR\Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


#Applying the Polynomial Regression Algorithm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))


#Applying the Support Vector Regression Algorithm
from sklearn.svm import SVR
regressor_2 = SVR(kernel="poly", degree=4, gamma="auto")
regressor_2.fit(x, y)
y_pred_2 = regressor_2.predict([[6.5]])


#Applying the K-Nearest Neighbors Algorithm
from sklearn.neighbors import KNeighborsRegressor
regressor_3 = KNeighborsRegressor(n_neighbors=2)
regressor_3.fit(x, y)
y_pred_3 = regressor_3.predict([[6.5]])


#Applying the Decision Tree Regression Algorithm
from sklearn.tree import DecisionTreeRegressor
regressor_4 = DecisionTreeRegressor()     
regressor_4.fit(x, y)
y_pred_4 = regressor_4.predict([[6.5]])


#Applying the Random Forest Regression Algorithm
from sklearn.ensemble import RandomForestRegressor
regressor_5 = RandomForestRegressor(n_estimators=129, criterion="absolute_error", min_samples_split=4)
regressor_5.fit(x, y)
y_pred_5 = regressor_5.predict([[6.5]])
