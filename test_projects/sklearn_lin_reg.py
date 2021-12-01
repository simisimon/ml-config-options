# Code source: Jaques Grobler
# License: BSD 3 clause

#import matplotlib.pyplot as plt
import numpy as np
import sklearn.base
from sklearn import datasets as ds, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import sklearn.base
import sklearn.base as b
from sklearn.base import BaseEstimator as BBB
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV as GC
from sklearn.model_selection import GroupShuffleSplit as gcc
from sklearn.model_selection import ParameterGrid, LeaveOneGroupOut as log
from sklearn.model_selection import GridSearchCV as GC2, GroupShuffleSplit
from sklearn.model_selection import GroupShuffleSplit as gcc2, GridSearchCV
import sklearn as     skl, tensorflow
import sklearn as kler
import sklearn



# Load the diabetes dataset
diabetes_X, diabetes_y = ds.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
#plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
#plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

#plt.xticks(())
#plt.yticks(())

#plt.show()