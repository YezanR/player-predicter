#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
import joblib
import numbers
import numpy as np
import sys
from data_transform import transformLearningData, extractFeatures

# Charger les données
df = pd.read_csv("data.csv")

df = transformLearningData(df)

features_df = extractFeatures(df)

joblib.dump(list(features_df.columns[:]), "features.csv")

# Create the X and y arrays
x = features_df.as_matrix()
y = df['Value'].as_matrix()

# on divise les données en données d'entrainement (70 %) et donnée de test (30 %)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Fit regression model
model = ensemble.GradientBoostingRegressor(
    n_estimators=1000,  # how many decision trees to build.
    learning_rate=0.1,  # how much each additional decision tree influences the overall prediction.
    max_depth=6,  # how many layers deep each individual decision tree can be.
    min_samples_leaf=9, # how many times a value must appear in our training set for a decision tree to make a decision based on it. 9 players must have the same characteristic before taken into consideration.
    max_features=0.1, # the percentage of features in our model that we randomly choose to consider each time we create a branch in our decision tree
    loss='huber',  # how scikit-learn calculates the model's error rate or cost as it learns.
)
# we tell the model to train using our training data set
model.fit(X_train, y_train)

# Save the trained model to a file so we can use it in other programs
joblib.dump(model, 'player_classifier_model.pkl')

# Find the error rate on the training set
mse = mean_absolute_error(y_train, model.predict(X_train))
print("Training Set Mean Absolute Error: %.4f" % mse)

# Find the error rate on the test set
mse = mean_absolute_error(y_test, model.predict(X_test))
print("Test Set Mean Absolute Error: %.4f" % mse)

print(model.predict(X_test))