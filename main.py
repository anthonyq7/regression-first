#STEP 1: IMPORTS

from sklearn.datasets import fetch_california_housing #built-in dataset with 20k houses and 8 features
from sklearn.model_selection import train_test_split #splits data into training and testing sets
from sklearn.linear_model import LinearRegression #the actual model class
from sklearn.metrics import mean_squared_error, r2_score #metric functions to evaluate how good our model is
import pandas as pd 
import numpy as np

#STEP 2: LOAD THE DATA
data = fetch_california_housing()
X = data.data #features (8 columns: income, house age, rooms, etc.)
y = data.target #target (house price in $100,000s)

print("Feature names: ", data.feature_names)
print("Number of samples: ", X.shape[0])
print("Number of features: ", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size= 0.2, #20% for testing, 80% for training
    random_state = 42 #Make results reproducable 
)

#STEP 3: CREATE AND TRAIN MODEL
model = LinearRegression()

#this is where the MAGIC happens -> it calculates BEST coefficients 
model.fit(X_train, y_train)

print("Model trained!")

#STEP 4: MAKE PREDICTIONS
y_pred = model.predict(X_test)

#looking at a few PREDICTIONS vs ACTUAL values

"""
print("\nFirst 5 predictions vs actual prices:")
for i in range(5):
    print(f"Predicted: ${y_pred[i]*100000:.0f}, Actual: ${y_test[i]*100000:.0f}")
"""


#STEP 5: EVALUATE MODEL
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:") #comments below are python f-string formate specifiers
print(f"RMSE: ${rmse*100000:.0f}") #:.0f is a float with 0 decimal places (rounded, no decimals) formatted as float
print(f"R^2 Score: {r2:.3f}") #.3f is a float with 3 decimal places formatted as float

#STEP 5: INSPECT WHAT MODEL LEARNED
print("\nFeature importance (coefficients):")
for name, coef in zip(data.feature_names, model.coef_):
    print(f"{name:15s}: {coef:+.4f}") #:+.4f always show sign (+ for positive, - for negative), 4 decimal places, float format
    #2.5678912 -> +2.5679
print(f"{'Intercept':15s}: {model.intercept_:+.4f}") #:15s is minimum width of 15 chars formatted as a string (adds spaces for padding)
