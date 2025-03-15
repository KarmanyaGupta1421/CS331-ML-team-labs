"""
Team: NaKaPr

"""


x_range = 5

a,b,c = map(int, input("Enter space separated values of a, b, c: ").split())


import numpy as np
import sklearn
import sklearn.model_selection
import matplotlib.pyplot as plt

def f1(x,a,b):
    return a*x + b + np.random.normal(0,np.sqrt(2)/5)

def f2(x,a,b,c):
    return a*x*x + b*x + c + np.random.normal(0,np.sqrt(2)/5)

def f3(x):
    return np.math.exp(x) + np.random.normal(0,np.sqrt(2)/5)

def generateDataset(a, b, c):
    D1 = []
    D2 = []
    D3 = []

    for _ in range(100):
        x = np.random.uniform(-x_range,x_range)
        D1.append([x,f1(x,a,b)])
        D2.append([x,f2(x,a,b,c)])
        D3.append([x,f3(x)])
    
    return D1,D2,D3

def least_square_error(X, Y, w):
    square_error = 0
    d = w.shape[0]
    for x,y in zip(X,Y):
        x_d = []
        for i in range(d):
            x_d.append(x**i)
        x_d = np.array(x_d)
        square_error += (y - w.T @ x_d)**2
    return square_error

def closed_linear_regression(D, d):
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(D.T[0], D.T[1], test_size = 0.8, random_state = 42, shuffle = True)
    A = []
    for x in X_train:
        x_d = []
        for i in range(d+1):
            x_d.append(x**i)
        A.append(x_d)
    
    A = np.array(A)
    w = (np.linalg.inv(A.T @ A) @ A.T) @ Y_train
    w = np.reshape(w, (d+1,1))

    train_LSE = least_square_error(X_train, Y_train, w)
    test_LSE = least_square_error(X_test, Y_test, w)
    # print(f"Train least square error for d = {d}:", train_LSE)
    # print(f"Test least square error for d = {d}:", test_LSE)

    return X_train, X_test, Y_train, Y_test, w

def plot_graphs(D, d):
    for i in d:
        X_train, X_test, Y_train, Y_test, w = closed_linear_regression(D, i)
        plt.scatter(X_test, Y_test)
        # plt.scatter(X_train, Y_train)
        xs = np.linspace(-5,5,100)
        ys = np.zeros(100)
        for j in range(i+1):
            ys += w[j] * (xs ** j)
        plt.plot(xs, ys)
    plt.show()

D1, D2, D3 = generateDataset(a,b,c)
D1 = np.array(D1)
D2 = np.array(D2)
D3 = np.array(D3)

X_train_closed, X_test_closed, Y_train_closed, Y_test_closed, w_closed_d1 = closed_linear_regression(D1, 1)
X_train_closed, X_test_closed, Y_train_closed, Y_test_closed, w_closed_d2 = closed_linear_regression(D2, 1)
X_train_closed, X_test_closed, Y_train_closed, Y_test_closed, w_closed_d3 = closed_linear_regression(D3, 1)

# plot_graphs(D1, [1,2,5,10])
# plot_graphs(D2, [1,2,5,10])
# plot_graphs(D3, [1,2,5,10])


def stocastic_gradient_descent(D, d):
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(D.T[0], D.T[1], test_size = 0.8, random_state = 42, shuffle = True)
    w_new = np.ones(shape=(d+1, 1))
    w_old = np.zeros(shape=(d+1, 1))
    gradient = np.ones(shape=(d+1, 1))
    epsilon = 1e-5
    learning_rate = 0.001
    overall_gradient = np.ones(shape = (d+1, 1))
    iters = 0
    while(iters < 4e3 and np.linalg.norm(overall_gradient) > epsilon):
        indices = np.arange(X_train.shape[0])
        i = np.random.choice(indices)
        xi = X_train[i]
        yi = Y_train[i]
        xi_d = []
        for j in range(d+1):
            xi_d.append(xi ** j)
        xi_d = np.array(xi_d)
        xi_d = np.reshape(xi_d, (d+1,1))
        yi = np.array(yi)
        yi = np.reshape(yi, (1,1))
        w_old = w_new
        gradient = (yi - w_old.T @ xi_d)[0] * xi_d
        w_new = w_old + learning_rate * gradient

        overall_gradient = 0
        for x,y in zip(X_train, Y_train):
            x_d = []
            for j in range(d+1):
                x_d.append(x ** j)
            x_d = np.array(x_d)
            x_d = np.reshape(x_d, (d+1,1))
            y = np.array(y)
            y = np.reshape(y, (1,1))
            overall_gradient += (y - w_new.T @ x_d)[0] * x_d
        overall_gradient *= (2/X_train.shape[0])

        iters += 1
        if (iters%1000 == 0):
            print(iters)

    print(np.linalg.norm(overall_gradient))
    return X_train, X_test, Y_train, Y_test, w_new

def plot_graphs_2(D, d, ind = 0):
    for i in d:
        X_train, X_test, Y_train, Y_test, w = stocastic_gradient_descent(D, i)
        print(w)
        # plt.scatter(X_test, Y_test)
        plt.scatter(X_train, Y_train)
        xs = np.linspace(-5,12,100)
        ys = np.zeros(100)
        for j in range(i+1):
            ys += w[j] * (xs ** j)
        plt.plot(xs, ys, label=f"d = {i}")
    plt.ylim(min(Y_test)-2,max(Y_test)+2)
    plt.xlim(min(X_test)-2,max(X_test)+2)
    plt.title(f"Stochastic Gradient Descent: Dataset {ind}")
    plt.legend()
    plt.show()

# plot_graphs_2(D1, [1,2,5,10], 1)
# plot_graphs_2(D2, [1,2,5,10], 2)
# plot_graphs_2(D3, [1,2,5,10], 3)


def ridge_regression(D, d, lamb = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]):
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(D.T[0], D.T[1], test_size = 0.8, random_state = 42, shuffle = True)

    X_validation, X_test, Y_validation, Y_test = sklearn.model_selection.train_test_split(X_test, Y_test, test_size = 0.75, random_state = 42, shuffle = True)
    A = []
    for x in X_train:
        x_d = []
        for i in range(d+1):
            x_d.append(x**i)
        A.append(x_d)

    best_lambda = None
    best_model = None
    best_val_error = float('inf')
    
    A = np.array(A)

    for l in lamb:
        w = (np.linalg.inv(A.T @ A + l * np.eye(d+1)) @ A.T) @ Y_train
        w = np.reshape(w, (d+1,1))
        val_error = least_square_error(X_validation, Y_validation, w)
        if val_error < best_val_error:
            best_val_error = val_error
            best_lambda = l
            best_model = w
        
    train_LSE = least_square_error(X_train, Y_train, best_model)
    test_LSE = least_square_error(X_test, Y_test, best_model)
    print(f"Train least square error for d = {d} and lambda = {best_lambda}:", train_LSE)
    print(f"Test least square error for d = {d} and lambda = {best_lambda}:", test_LSE)

    return X_train, X_test, Y_train, Y_test, best_model, best_lambda, least_square_error(X_test, Y_test, best_model)

dataset_num = 0
def plot_graphs_3(D, d):
    global dataset_num
    dataset_num += 1
    best_d = None
    best_model = None
    best_error = float('inf')

    for i in d:
        X_train, X_test, Y_train, Y_test, w, lamb, error = ridge_regression(D, i)
        if(error < best_error):
            best_error = error
            best_d = i
            best_model = w
        plt.scatter(X_test, Y_test)
        # plt.scatter(X_train, Y_train)
        xs = np.linspace(-5,5,100)
        ys = np.zeros(100)
        for j in range(i+1):
            ys += w[j] * (xs ** j)
        plt.plot(xs, ys)
    plt.ylim(min(Y_test)-2,max(Y_test)+2)
    plt.xlim(min(X_test)-2,max(X_test)+2)
    plt.show()

    print(f"Best d for dataset: {best_d}")
    X_train_closed, X_test_closed, Y_train_closed, Y_test_closed, w_closed = closed_linear_regression(D, best_d)
    # print("Dataset", dataset_num, ": best model:", best_model)
    # print("Dataset", dataset_num, ": closed model:", w_closed)

    
plot_graphs_3(D1, [1,2,5,10])
plot_graphs_3(D2, [1,2,5,10])
plot_graphs_3(D3, [1,2,5,10])
