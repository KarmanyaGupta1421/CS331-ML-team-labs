"""
Team: NaKaPr
Rolls: 2203311 , 2203319 , 2203121
"""

import numpy as np

def multivariate_gaussian(X, mu, cov):
    d = mu.shape[0]
    if cov.ndim == 1:
        cov = np.diag(cov)
        
    X = X - mu
    p = (2* np.pi)**(-d/2) * np.linalg.det(cov)**(-0.5) * np.exp(-0.5 * (X.T @ np.linalg.pinv(cov) @ X))
    
    return p
    
p0 = 0.5
p1 = 0.5

n = int(input("n: "))
mu0 = np.array(list(map(int, input("Enter mu0: ").split()))[:3])
cov0 = np.array([list(map(int, input(f"Enter {i+1}th row of cov0: ").split()))[:3] for i in range (3)])

mu1 = np.array(list(map(int, input("Enter mu1: ").split()))[:3])
cov1 = np.array([list(map(int, input(f"Enter {i+1}th row of cov1: ").split()))[:3] for i in range (3)])

# mu0 = np.array([1,1,1])
# cov0 = np.array([[1,1,1],[1,1,1],[1,1,1]])

# mu0 = np.array([1,2,3])
# cov0 = np.array([[1,0,0],[0,2,0],[0,0,3]])

# mu1 = np.array([1,1,1])
# cov1 = np.array([[1,-3,0.5],[-3,9.3333,-1.6667],[0.5,-1.6667,0.3333]])

# mu1 = np.array([4,5,6])
# cov1 = np.array([[4,0,0],[0,5,0],[0,0,6]])

Y = np.random.binomial(1, p1, n)
X = []

for elem in Y:
    if (elem == 0):
        X.append(np.random.multivariate_normal(mu0, cov0))
    else:
        X.append(np.random.multivariate_normal(mu1, cov1))

X = np.array(X)

def single_feature(Y, X, mu0, cov0, mu1, cov1):
    n = Y.shape[0]
    loss = []
    for i in range (3):
        error_cnt = 0
        for j in range (n):
            x = np.array([X[j][i]])
            Mu = np.array([mu0[i]])
            Cov = np.array([cov0[i][i]])

            p0f0 = p0 * multivariate_gaussian(x, Mu, Cov)

            Mu = np.array([mu1[i]])
            Cov = np.array([cov1[i][i]])

            p1f1 = p1 * multivariate_gaussian(x, Mu, Cov)

            if (p0f0 >= p1f1 and Y[j] == 1):
                error_cnt += 1
            elif (p1f1 > p0f0 and Y[j] == 0):
                error_cnt += 1
        
        loss.append(error_cnt/n)
    
    return loss

def double_feature(Y, X, mu0, cov0, mu1, cov1):
    n = Y.shape[0]
    loss = []

    for i in range(3):
        for j in range (i+1, 3):
            error_cnt = 0
            for k in range(n):
                x = np.array([X[k][i], X[k][j]])
                Mu = np.array([mu0[i], mu0[j]])
                Cov = []

                if (i == 0 and j == 1):
                    Cov = np.array([[cov0[0][0], cov0[0][1]], [cov0[1][0], cov0[1][1]]])
                elif(i == 0 and j == 2):
                    Cov = np.array([[cov0[0][0], cov0[0][2]], [cov0[2][0], cov0[2][2]]])
                else:
                    Cov = np.array([[cov0[1][1], cov0[1][2]], [cov0[2][1], cov0[2][2]]])

                p0f0 = p0 * multivariate_gaussian(x, Mu, Cov)

                Mu = np.array([mu1[i], mu1[j]])
                Cov = []

                if (i == 0 and j == 1):
                    Cov = np.array([[cov1[0][0], cov1[0][1]], [cov1[1][0], cov1[1][1]]])
                elif(i == 0 and j == 2):
                    Cov = np.array([[cov1[0][0], cov1[0][2]], [cov1[2][0], cov1[2][2]]])
                else:
                    Cov = np.array([[cov1[1][1], cov1[1][2]], [cov1[2][1], cov1[2][2]]])

                p1f1 = p1 * multivariate_gaussian(x, Mu, Cov)

                if (p0f0 >= p1f1 and Y[k] == 1):
                    error_cnt += 1
                elif (p1f1 > p0f0 and Y[k] == 0):
                    error_cnt += 1
            
            loss.append(error_cnt/n)

    return loss

def triple_feature(Y, X, mu0, cov0, mu1, cov1):
    n = Y.shape[0]
    loss = []

    error_cnt = 0
    
    for i in range(n):
        p0f0 = p0 * multivariate_gaussian(X[i] , mu0 , cov0)
        p1f1 = p1 * multivariate_gaussian(X[i] , mu1 , cov1)

        if (p0f0 >= p1f1 and Y[i] == 1):
            error_cnt += 1
        elif (p1f1 > p0f0 and Y[i] == 0):
            error_cnt += 1

    loss.append(error_cnt/n)
    return loss
            


single_feature_loss = single_feature(Y, X, mu0, cov0, mu1, cov1)
double_feature_loss = double_feature(Y, X, mu0, cov0, mu1, cov1)
triple_feature_loss = triple_feature(Y, X, mu0, cov0, mu1, cov1)
print("Single feature losses:",single_feature_loss)
print("Double feature losses:",double_feature_loss)
print("Triple feature loss:",triple_feature_loss)
    
# we observe that as we take more featutres into consideration, out overall error decreases

def single_feature_mean_classifier(Y , X , mu0 , mu1):
    n = Y.shape[0]
    loss = []
    for i in range(3):
        error_cnt = 0
        for j in range(n):
            distmu0 = abs(mu0[i] - X[j][i])
            distmu1 = abs(mu1[i] - X[j][i])
            if(distmu0 >= distmu1 and Y[j] == 0):
                error_cnt += 1
            elif(distmu0 < distmu1 and Y[j] == 1):
                error_cnt += 1
        loss.append(error_cnt/n)
    
    return loss

def double_feature_mean_classifier(Y , X , mu0 , mu1):
    n = Y.shape[0]
    loss = []
    for i in range(3):
        for j in range(i+1 , 3):
            error_cnt = 0
            for k in range(n):
                distmu0 = ((mu0[i] - X[k][i])**2 + (mu0[j] - X[k][j])**2)**0.5
                distmu1 = ((mu1[i] - X[k][i])**2 + (mu1[j] - X[k][j])**2)**0.5
                if(distmu0 >= distmu1 and Y[k] == 0):
                    error_cnt += 1
                elif(distmu0 < distmu1 and Y[k] == 1):
                    error_cnt += 1
            loss.append(error_cnt/n)
    
    return loss

def triple_feature_mean_classifier(Y , X , mu0 , mu1):
    n = Y.shape[0]
    loss = []

    error_cnt = 0
    
    for i in range(n):
        distmu0 = np.linalg.norm(X[i] - mu0)
        distmu1 = np.linalg.norm(X[i] - mu1)

        if(distmu0 >= distmu1 and Y[i] == 0):
            error_cnt += 1
        elif(distmu0 < distmu1 and Y[i] == 1):
            error_cnt += 1
    loss.append(error_cnt/n)
    return loss


single_feature_classifier_loss = single_feature_mean_classifier(Y , X , mu0 , mu1)
double_feature_classifier_loss = double_feature_mean_classifier(Y , X , mu0 , mu1)
triple_feature_classifier_loss = triple_feature_mean_classifier(Y , X , mu0 , mu1)

print("Single feature mean classifier losses:",single_feature_classifier_loss)
print("Double feature mean classifier losses:",double_feature_classifier_loss)
print("Triple feature mean classifier loss:",triple_feature_classifier_loss)