import numpy as np
import matplotlib.pyplot as plt
import time

class Kmeans():
    def __init__(self, num_classes, max_iters, random_state = 123):
        self.clusters = num_classes
        self.max_iters = max_iters
        self.random_seed = random_state
        
    def initialize_centroids(self, X_data):
        np.random.RandomState(self.random_seed)
        random_idx = np.random.permutation(X_data.shape[0])
        centroids = X_data[random_idx[:self.clusters]]
        return centroids
    
    # def calculate_new_centroids(self, distances):
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
    
    def train(self, X_data):
        self.centroids = self.initialize_centroids(X_data)
        for i in range(self.max_iters):
            old_centroids = self.centroids
            distances = self.calculate_distance(X_data, self.centroids)
            self.labels = self.find_closest_cluster(distances)
            self.centroids = self.calculate_new_centroids(X_data, self.labels)
            if np.all(old_centroids == self.centroids):
                break
        self.error = self.compute_sse(X_data, self.labels, self.centroids)

    def pca(self, X_data):
        covariance_matrix = np.cov(X_data.T)
        eigvals, eigvecs =  np.linalg.eig(covariance_matrix)
        pca = np.sort(eigvals.real/np.sum(eigvals))[::-1]
        plt.bar([f'pc{i}' for i in range(1, len(pca) + 1)], pca.real)

        named_tuple = time.localtime() # get struct_time
        time_string = time.strftime("(%m-%d-%Y-%H:%M:%S)", named_tuple)

        plt.savefig(f'results/kmeans_pca{time_string}')







    