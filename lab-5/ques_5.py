'''
Team Name - NaKaPr
'''

import numpy as np
import sklearn
from sklearn.datasets import load_digits
import sklearn.model_selection
import matplotlib.pyplot as plt

def transform_to_binary(data):
    for img in range(len(data.images)):
        for i in range(len(data.images[img])):
            for j in range(len(data.images[img][i])):
                if(data.images[img][i][j] > 0):
                    data.images[img][i][j] = 1
                else:
                    data.images[img][i][j] = 0
    return data
 

def train_naive_bayes(X_train, Y_train):
    feature_count_array = np.zeros(shape = (64, 2, 10))
    y_count = np.zeros(shape = (10,))

    for (image , label) in zip(X_train, Y_train):
        y_count[label] += 1 
        flat_image = image.flatten()
        for i in range(len(flat_image)):
            feature_count_array[i][flat_image[i]][label] += 1
    
    return feature_count_array, y_count


def p_xi_y(feature_count_array, y_count, a, b, i , xi , y):
    num = feature_count_array[i][xi][y] + a
    den = y_count[y] + a + b
    return num / den 

def loss_01(feature_count_array, y_count, X_test, Y_test, a, b):
    """
    Evaluate the classifier on Dtest and report empirical 0-1 loss.
    """
    loss = 0

    for image, label in zip(X_test, Y_test):
        flat_image = image.flatten()
        prob_array = np.zeros(shape = (10,))
        for y in range(10):
            prob = 1
            for ind, pixel in enumerate(flat_image):
                prob *= p_xi_y(feature_count_array, y_count, a, b, ind, pixel, y)
            prob_array[y] = prob * (y_count[y] / sum(y_count))

        prob_array /= sum(prob_array)

        prediction = np.argmax(prob_array, axis=0)
        if(prediction != label):
            loss += 1
    
    return loss / Y_test.shape[0]


mnist = load_digits()
mnist = transform_to_binary(mnist)
mnist.images = np.array(mnist.images, dtype=int)

splits = 5
kf = sklearn.model_selection.KFold(n_splits = splits, shuffle = True, random_state=42)

values = [1, 5, 10, 20, 80, 100, 1000]
loss = np.zeros(shape = (len(values), len(values)))

for i in range(len(values)):
    for j in range(len(values)):
        a = values[i]
        b = values[j]

        avg_loss = 0
        for train_index, test_index in kf.split(mnist.images):
            X_train, X_test = mnist.images[train_index], mnist.images[test_index]
            Y_train, Y_test = mnist.target[train_index], mnist.target[test_index]

            feature_count_array, y_count = train_naive_bayes(X_train, Y_train)
            curr_loss = loss_01(feature_count_array, y_count, X_test, Y_test, a, b)
            avg_loss += curr_loss
        
        avg_loss /= splits

        loss[i][j] = avg_loss

        print(f"a: {a}, b: {b}, avg_loss: {avg_loss}")

opt_a = opt_b = -1
opt_loss = 100

for i in range(len(values)):
    for j in range(len(values)):
        if (loss[i][j] < opt_loss):
            opt_loss = loss[i][j]
            opt_a = values[i]
            opt_b = values[j]

print()
print("Minimum loss: ", opt_loss)
print("optimal a: ", opt_a)
print("optimal b: ", opt_b)