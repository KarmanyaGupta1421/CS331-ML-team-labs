"""
Team name: NaKaPr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def multivariate_gaussian(X, mu, cov):
    d = mu.shape[0]
    if cov.ndim == 1:
        cov = np.diag(cov)
    X = X - mu
    X = X.T
    p = (2* np.pi)**(-d/2) * np.linalg.det(cov)**(-0.5) * np.exp(-0.5 * (X.T @ np.linalg.pinv(cov) @ X))
    
    return p

def log_likelihood(X, Mu, Cov, Pi):
    n = X.shape[0]
    d = X.shape[1]
    k = Pi.shape[0]

    log_lik = 0
    for i in range(n):
        sum = 0
        for j in range(k):
            fi = multivariate_gaussian(X[i], Mu[j], Cov[j])
            sum += Pi[j] * fi
        log_lik += np.math.log(sum)
    return log_lik

def EM_algorithm(X, k, Mu_old, Cov_old):
    n = X.shape[0]
    d = X.shape[2]
    Pi_old = np.ones(shape=(k))
    Pi_old /= k

    prev_log_like = log_likelihood(X, Mu_old, Cov_old, Pi_old)+1
    new_log_like = log_likelihood(X, Mu_old, Cov_old, Pi_old)
    
    log_array = [new_log_like]

    while(abs(prev_log_like - new_log_like) >= 1e-6):
        
        # E-step
        gamma = np.zeros(shape = (n,k))
        for i in range (n):
            den = 0
            for j in range(k):
                den += Pi_old[j] * multivariate_gaussian(X[i], Mu_old[j], Cov_old[j])
            for j in range(k):
                num = Pi_old[j] * multivariate_gaussian(X[i], Mu_old[j], Cov_old[j])
                gamma[i][j] = num / den 

        # M-step
        Pi_new = np.ones(shape=(k))
        Mu_new = np.zeros(shape=(k,d))
        Cov_new = np.zeros(shape = (k,d,d))

        for i in range(k):
            N_i = np.sum(gamma.T[i])
            
            # mu_i_new
            sum_mu = np.zeros(shape = (1,d))
            for j in range(n):
                sum_mu += gamma[j][i] * X[j]
            sum_mu /= N_i
            Mu_new[i] = sum_mu

            # cov_i_new
            sum_cov = np.zeros(shape = (1,d,d))
            for j in range(n):
                a = gamma[j][i] * ((X[j] - Mu_new[i]).T @ (X[j] - Mu_new[i]))
                sum_cov += gamma[j][i] * ((X[j] - Mu_new[i]).T @ (X[j] - Mu_new[i]))
            sum_cov /= N_i
            Cov_new[i] = sum_cov

            # mu_k_new
            Pi_new[i] = N_i / n
        
        Mu_old = Mu_new
        Cov_old = Cov_new
        Pi_old = Pi_new

        prev_log_like = new_log_like
        new_log_like = log_likelihood(X, Mu_new, Cov_new, Pi_new)
        log_array.append(new_log_like)
    
    cluster_array = np.zeros(shape = (n, k))
    for i in range(n):
        cluster_array[i][np.argmax(gamma[i])] = 1
    
    
    return cluster_array, log_array, Mu_new


def get_results(file_name, Mu_old, Cov_old):
    df = pd.read_csv(file_name)
    df[df.columns[1]] = (df[df.columns[1]] - df[df.columns[1]].mean()) / df[df.columns[1]].std()
    df[df.columns[2]] = (df[df.columns[2]] - df[df.columns[2]].mean()) / df[df.columns[2]].std()

    X = []
    for _, row in df.iterrows():
            X.append([row[df.columns[1]], row[df.columns[2]]])
    X = np.array(X)
    X = X.reshape((-1,1,2))

    cluster, log_array, Mu = EM_algorithm(X, 2, Mu_old, Cov_old)

    return cluster, log_array, Mu


Mu_faithful_old = np.array([[[1,-1]],[[-1,1]]])
Mu_dataset_old = np.array([[[2,-3]],[[2,3]]])
Cov_old = np.array([[[1,0],[0,1]], [[1,0],[0,1]]])

faithful_cluster, faithful_log, Mu_new_faithful = get_results("faithful.csv", Mu_faithful_old, Cov_old)
dataset_cluster, dataset_log, Mu_new_dataset = get_results("dataset.csv", Mu_dataset_old, Cov_old)


fig, ax = plt.subplots(1,2)

ax[0].plot(faithful_log)
ax[0].set_title("Log Likelihood faithful.csv")
ax[0].set_xlabel("iteration no.")
ax[0].set_ylabel("Log Likelihood")

ax[1].plot(dataset_log)
ax[1].set_title("Log Likelihood dataset.csv")
ax[1].set_xlabel("iteration no.")
ax[1].set_ylabel("Log Likelihood")
plt.tight_layout()
plt.show()



def get_plots_of_clusters(file_name, Mu, Cov, final_cluster, Mu_new):
    df = pd.read_csv(file_name)
    df[df.columns[1]] = (df[df.columns[1]] - df[df.columns[1]].mean()) / df[df.columns[1]].std()
    df[df.columns[2]] = (df[df.columns[2]] - df[df.columns[2]].mean()) / df[df.columns[2]].std()

    X = []
    for _, row in df.iterrows():
            X.append([row[df.columns[1]], row[df.columns[2]]])
    X = np.array(X)
    X = X.reshape((-1,1,2))

    n = X.shape[0]
    k = X.shape[2]

    Pi = np.ones(shape=(k))
    Pi /= k


    gamma = np.zeros(shape = (n,k))
    for i in range (n):
        den = 0
        for j in range(k):
            den += Pi[j] * multivariate_gaussian(X[i], Mu[j], Cov[j])
        for j in range(k):
            num = Pi[j] * multivariate_gaussian(X[i], Mu[j], Cov[j])
            gamma[i][j] = num / den 

    cluster_array = np.zeros(shape = (n, k))
    for i in range(n):
        cluster_array[i][np.argmax(gamma[i])] = 1

    fig, ax = plt.subplots(1, 2)

    green_X = []
    green_Y = []
    blue_X = []
    blue_Y = []
    
    for i in range(n):
        if(cluster_array[i][0] == 1):
            green_X.append(X[i][0][0])
            green_Y.append(X[i][0][1])
        else:
            blue_X.append(X[i][0][0])
            blue_Y.append(X[i][0][1])
            
    ax[0].scatter(green_X, green_Y, marker='.', color="lightgreen")
    ax[0].scatter(blue_X, blue_Y, marker='.', color="lightblue")
    ax[0].scatter(Mu[0][0][0], Mu[0][0][1], color = "green", label = "center 1", marker='X')
    ax[0].scatter(Mu[1][0][0], Mu[1][0][1], color = "blue", label = "center 2", marker='X')
    ax[0].set_title("Old Clusters")
    
    green_X = []
    green_Y = []
    blue_X = []
    blue_Y = []
    
    for i in range(n):
        if(final_cluster[i][0] == 1):
            green_X.append(X[i][0][0])
            green_Y.append(X[i][0][1])
        else:
            blue_X.append(X[i][0][0])
            blue_Y.append(X[i][0][1])

    ax.flatten()[1].scatter(green_X, green_Y, marker='.', color="lightgreen")
    ax.flatten()[1].scatter(blue_X, blue_Y, marker='.', color="lightblue")
    ax.flatten()[1].scatter(Mu_new[0][0], Mu_new[0][1], color = "green", label = "center 1", marker='X')
    ax.flatten()[1].scatter(Mu_new[1][0], Mu_new[1][1], color = "blue", label = "center 2", marker='X')
    ax.flatten()[1].set_title("New Clusters")
    
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc = "lower right")
    plt.tight_layout()
    fig.suptitle(f"Initial and final clusters for {file_name}")
    plt.show()

get_plots_of_clusters("faithful.csv", Mu_faithful_old, Cov_old, final_cluster=faithful_cluster, Mu_new=Mu_new_faithful)
get_plots_of_clusters("dataset.csv", Mu_dataset_old, Cov_old, final_cluster=dataset_cluster, Mu_new=Mu_new_dataset)

