'''
Team Name - NaKaPr
'''

import numpy as np
import sklearn
from sklearn.datasets import load_digits
import sklearn.model_selection

# Loading MNIST dataset
mnist = load_digits()


"""
Transform all features to binary using the following operation:
x_new = 1 [xi > 0]
"""

def transform_to_binary():
    for img in range(len(mnist.images)):
        for i in range(len(mnist.images[img])):
            for j in range(len(mnist.images[img][i])):
                if(mnist.images[img][i][j] > 0):
                    mnist.images[img][i][j] = 1
                else:
                    mnist.images[img][i][j] = 0
            
mnist.images = np.array(mnist.images, dtype=int) # converting to integers rather than floats
transform_to_binary()

"""
Perform a split of the dataset into a training set Dtrain (80% of points) and test set
Dtest (remaining 20% of points).
"""

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(mnist.images, mnist.target, test_size = 0.2, random_state = 42, shuffle = True)

"""
Using Dtrain learn a Naive Bayes classifier with smoothing incorporated- use a = b = 1 for the hyper-parameters (smoothing).
"""

def p_xi_y(i , xi , y):
    a = b = 1
    num = feature_count_array[i][xi][y] + a
    den = y_count[y] + a + b
    return num / den  

def train_naive_bayes():
    feature_count_array = np.zeros(shape = (64, 2, 10))
    y_count = np.zeros(shape = (10,))

    for (image , label) in zip(X_train, Y_train):
        y_count[label] += 1 
        flat_image = image.flatten()
        for i in range(len(flat_image)):
            feature_count_array[i][flat_image[i]][label] += 1
    
    return feature_count_array, y_count

feature_count_array, y_count = train_naive_bayes()

"""
Evaluate the classifier on Dtest and report empirical 0-1 loss.
"""

def loss_01():
    loss = 0

    for image, label in zip(X_test, Y_test):
        flat_image = image.flatten()
        prob_array = np.zeros(shape = (10,))
        for y in range(10):
            prob = 1
            for ind, pixel in enumerate(flat_image):
                prob *= p_xi_y(ind, pixel, y)
            prob_array[y] = prob * (y_count[y] / sum(y_count))

        prob_array /= sum(prob_array)

        prediction = np.argmax(prob_array, axis=0)
        if(prediction != label):
            loss += 1
    
    return loss / Y_test.shape[0]

loss = loss_01()

print("loss =", loss)