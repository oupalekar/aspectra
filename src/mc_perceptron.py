import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from visualize import *


class MultiClassPerceptron():
    def __init__(self, visualize = False):
         self.weights = None
         self.visualize = visualize

    def load_model(self, filepath):
        self.weights = np.load(filepath)
    
    def save_model(self, filepath):
        np.save(filepath, self.weights)

    def train(self, X_train, Y_train, epochs, lr=0.01):
        epoch_values = []
        error_values = []

        self.weights = np.zeros((X_train.shape[1], Y_train.shape[1]))
        for k in range(epochs):
            misclassified = 0
            for x, y in zip(X_train, Y_train):
                y_hat = np.argmax(np.sign(np.dot(self.weights.T, x)))
                y = np.argmax(y)
                if y_hat != y:
                    misclassified += 1
                    self.weights[:, y_hat] -= lr * x
                    self.weights[:, y] += lr * x
            error_values.append(misclassified/X_train.shape[0])
            epoch_values.append(k)
            
        error_vs_epochs(error_values, epoch_values, None) if self.visualize else None

            

    def predict_class(self, X_predict, Y_predict):
        predicted_class = np.zeros((X_predict.shape[0], Y_predict.shape[1]))
        predictions = np.dot(self.weights.T, X_predict.T).T
        for i in range(X_predict.shape[0]):
            prediction = predictions[i]
            for j in range(len(prediction)):
                guess = prediction[j]
                if guess > 0:
                    predicted_class[i][j] = 1
                    break
        return predicted_class

    def accuracy(self, predicted, expected):
        error = 0
        numsamples = expected.shape[0]
        for i in range(numsamples):
            if np.array_equal(predicted[i], expected[i]):
                    continue
            else:   
                    error += 1
        return (1-error/numsamples)