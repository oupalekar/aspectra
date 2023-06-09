import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

class Kmeans():
    def __init__(self, num_classes, max_iters, random_state = 123):
        self.clusters = num_classes
        self.max_iters = max_iters
        self.random_seed = random_state
        
    def initialize_centroids(self, X_data, Y_data=None):
        X_df = pd.DataFrame(X_data)
        features = X_data.shape[1]
        Y_df = pd.DataFrame(Y_data)
        combined_df = pd.concat([X_df, Y_df], axis = 1, ignore_index=True)
        centroids = np.zeros((self.clusters, features))
        np.random.RandomState(self.random_seed)
        for i in range(self.clusters):
            curr_df = combined_df[combined_df[features + i] == 1.0]
            random_row = curr_df.sample(1).to_numpy()[0][:features]
            # print(random_row)
            centroids[i, :] = random_row
        return centroids
    
    def calculate_distance(self, X_data, centroids):
        distance = np.zeros((X_data.shape[0], self.clusters))
        for k in range(self.clusters):
            row_norm = np.linalg.norm(X_data - centroids[k, :], axis = 1)
            distance[:, k] = np.square(row_norm)
        return distance
    
    def find_closest_cluster(self, distances):
        return np.argmin(distances, axis = 1)
    
    def calculate_new_centroids(self, X_data, labels):
        centroids = np.zeros((self.clusters, X_data.shape[1]))
        for i in range(self.clusters):
            centroids[i, :] = np.mean(X_data[labels == i, :], axis=0)  
        return centroids  
    
    def compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.clusters):
            distance[labels == k] = np.linalg.norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))
    
    def train(self, X_data, Y_data):
        self.centroids = self.initialize_centroids(X_data, Y_data)
        self.centroids = np.array([[-1.2, -0.4, -1.0], [-0.6, 0.0, -1.0], [0.0,0.4, -1.0]])
        print(self.centroids)
        for i in range(self.max_iters):
            old_centroids = self.centroids
            distances = self.calculate_distance(X_data, self.centroids)
            self.labels = self.find_closest_cluster(distances)
            self.centroids = self.calculate_new_centroids(X_data, self.labels)
            if np.all(old_centroids == self.centroids):
                print(f"Breaking at i = {i}")
                break
        self.error = self.compute_sse(X_data, self.labels, self.centroids)

    def pca(self, X_data, plot=False):
        covariance_matrix = np.cov(X_data.T)
        eigvals, eigvecs =  np.linalg.eig(covariance_matrix)
        pca = eigvals.real/np.sum(eigvals)
        if plot:
            plt.bar([f'pc{i}' for i in range(1, len(pca) + 1)], pca.real)

            named_tuple = time.localtime() # get struct_time
            time_string = time.strftime("(%m-%d-%Y-%H:%M:%S)", named_tuple)

            plt.savefig(f'results/kmeans_pca{time_string}')

        pca_count = len(np.cumsum(pca)[np.cumsum(pca) < 0.95])
        print(pca_count)
        pca_vec = eigvecs[:, 0:pca_count]
        return pca_vec







    