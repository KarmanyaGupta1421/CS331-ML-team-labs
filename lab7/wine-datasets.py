"""
Team: NaKaPr

"""

import pandas as pd
import numpy as np
# import sklearn
import sklearn.model_selection
import matplotlib.pyplot as plt

red_df = pd.read_csv("winequality-red.csv", sep = ";")
white_df = pd.read_csv("winequality-white.csv", sep = ";")

# X and Y for red and white wine datasets
X_red = red_df.iloc[:,:-1].values
Y_red = red_df.iloc[:,-1].values
X_white = white_df.iloc[:,:-1].values
Y_white = white_df.iloc[:,-1].values

num_features = X_red.shape[1]
# print(num_features)

def least_square_error(X, Y, w):
    square_error = 0
    d = w.shape[0]
    for x,y in zip(X,Y):
        x_d = [1]
        for feature in x:
            for i in range(1, (d-1)//num_features+1):
                x_d.append(feature**i)
        x_d = np.array(x_d)
        x_d = np.reshape(x_d, (x_d.shape[0], 1))
        # print(w.T.shape, x_d.shape)
        # break
        square_error += ((y - w.T @ x_d))**2
    return square_error

# def ridge_regression(X, Y, d, lamb = np.linspace(1e-4, 1e4, 100)):
def ridge_regression(X, Y, d, lamb = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]):
    # X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.6, random_state = 42, shuffle = True)
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.6, shuffle = True)

    # X_validation, X_test, Y_validation, Y_test = sklearn.model_selection.train_test_split(X_test, Y_test, test_size = 2/3, random_state = 42, shuffle = True)
    X_validation, X_test, Y_validation, Y_test = sklearn.model_selection.train_test_split(X_test, Y_test, test_size = 2/3, shuffle = True)

    A = []
    for x in X_train:
        x_d = [1]
        for feature in x:
            for i in range(1, d+1):
                x_d.append(feature**i)
        A.append(x_d)

    best_lambda = None
    best_model = None
    best_val_error = float('inf')
    
    A = np.array(A)

    for l in lamb:
        w = (np.linalg.inv(A.T @ A + l * np.eye(num_features*(d)+1)) @ A.T) @ Y_train
        w = np.reshape(w, (num_features*(d)+1,1))
        val_error = least_square_error(X_validation, Y_validation, w)
        # print(val_error, best_val_error)
        # print()
        if val_error < best_val_error:
            best_val_error = val_error
            best_lambda = l
            best_model = w
        
    train_LSE = least_square_error(X_train, Y_train, best_model)
    test_LSE = least_square_error(X_test, Y_test, best_model)
    # print(f"Train least square error for d = {d} and lambda = {best_lambda}:", train_LSE)
    # print(f"Test least square error for d = {d} and lambda = {best_lambda}:", test_LSE)

    return X_train, X_test, Y_train, Y_test, best_model, best_lambda, least_square_error(X_test, Y_test, best_model)


# RED WINE DATASET PREDICTION

print("RED WINE DATASET PREDICTION\n")

D = None
LAMBDA = None
ERROR = float('inf')

for d in range(1, 11):
    X_train, X_test, Y_train, Y_test, best_model, best_lambda, error = ridge_regression(X_red, Y_red, d)
    print(f"Test least square error for d = {d} and lambda = {best_lambda}:", error)
    if error < ERROR:
        ERROR = error
        D = d
        LAMBDA = best_lambda

print("\nBest d:", D)
print("Best lambda:", LAMBDA, end = "\n\n")

# WHITE WINE DATASET PREDICTION

print("WHITE WINE DATASET PREDICTION\n")

D = None
LAMBDA = None
ERROR = float('inf')

for d in range(1, 11):
    X_train, X_test, Y_train, Y_test, best_model, best_lambda, error = ridge_regression(X_white, Y_white, d)
    print(f"Test least square error for d = {d} and lambda = {best_lambda}:", error)
    if error < ERROR:
        ERROR = error
        D = d
        LAMBDA = best_lambda

print("\nBest d:", D)
print("Best lambda:", LAMBDA, end = "\n\n")
