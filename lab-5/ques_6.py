'''
Team Name - NaKaPr
'''

import numpy as np
import sklearn
from sklearn.datasets import load_digits
import sklearn.model_selection

def transform_to_binary(data):
    for img in range(len(data.images)):
        for i in range(len(data.images[img])):
            for j in range(len(data.images[img][i])):
                if(data.images[img][i][j] > 0):
                    data.images[img][i][j] = 1
                else:
                    data.images[img][i][j] = 0
    return data


mnist = load_digits()
mnist = transform_to_binary(mnist)
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(mnist.data, mnist.target, test_size = 0.2, random_state = 42, shuffle = True)
X_train = np.array([img.flatten() for img in X_train])
X_test = np.array([img.flatten() for img in X_test])

def knn_classifier(X_train, X_test, Y_train, Y_test, k):
    loss = 0
    for image, label in zip(X_test, Y_test):
        nearest = []
        for train_image, train_label in zip(X_train, Y_train):
            dist = np.linalg.norm(image - train_image)
            nearest.append([dist, train_label])
        
        nearest.sort()

        count_array = np.zeros(10)

        for elem in nearest[:k]:
            count_array[elem[1]] += 1
        
        prediction = np.argmax(count_array)

        if (prediction != label):
            loss += 1

    return loss/Y_test.shape[0]

loss_3 = knn_classifier(X_train, X_test, Y_train, Y_test, 3)
loss_5 = knn_classifier(X_train, X_test, Y_train, Y_test, 5)

print("Loss for k = 3 :", loss_3)
print("Loss for k = 5 :", loss_5)
            