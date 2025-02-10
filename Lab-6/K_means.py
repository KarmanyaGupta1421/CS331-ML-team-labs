"""
Team name: NaKaPr
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_dist(p1, p2):
    dist = np.math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return dist



def k_means(filename, Mu):
    x_c1 = []
    y_c1 = []
    x_c2 = []
    y_c2 = []
    new_c1 = Mu[0]
    new_c2 = Mu[1]
    prev_c1 = [0, 0]
    prev_c2 = [0, 0]

    df = pd.read_csv(filename)
    df[df.columns[1]] = (df[df.columns[1]] - df[df.columns[1]].mean()) / df[df.columns[1]].std()
    df[df.columns[2]] = (df[df.columns[2]] - df[df.columns[2]].mean()) / df[df.columns[2]].std()

    while(new_c1 != prev_c1 or new_c2 != prev_c2):
        cluster1 = []
        cluster2 = []
        j1 = 0
        j2 = 0

        # assign clusters
        for _, row in df.iterrows():
            curr = [row[df.columns[1]], row[df.columns[2]]]
            d1 = get_dist(new_c1, curr)
            d2 = get_dist(new_c2, curr)

            if (d1 < d2):
                cluster1.append(curr)
                j1 += d1*d1
            else:
                cluster2.append(curr)
                j1 += d2*d2
        
        # calculate new centers
        prev_c1 = new_c1
        prev_c2 = new_c2

        new_c1 = list(np.mean(cluster1, axis = 0))
        new_c2 = list(np.mean(cluster2, axis = 0))

        # plot graph
        x_c1 = [item[0] for item in cluster1]
        y_c1 = [item[1] for item in cluster1]
        x_c2 = [item[0] for item in cluster2]
        y_c2 = [item[1] for item in cluster2]

    plt.scatter(x_c1, y_c1, color = "lightgreen", label = "cluster 1")
    plt.scatter(x_c2, y_c2, color = "lightblue", label = "cluster 2")
    plt.scatter(prev_c1[0], prev_c1[1], color = "green", label = "center 1", marker='X')
    plt.scatter(prev_c2[0], prev_c2[1], color = "blue", label = "center 2", marker='X')
    plt.title(f"K-means final clusters for {filename}")
    plt.legend()
    plt.show()



k_means("faithful.csv", [[1, -1], [-1, 1]])
k_means("dataset.csv", [[2, -3], [2, 3]])