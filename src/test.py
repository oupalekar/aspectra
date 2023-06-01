import sys
import mc_perceptron
import utils
import parse
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from kmeans import *
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans


## ML Perceptron ##
def ml_perceptron(visualize = False):
    X_df, Y_df = parse.preprocess()
    scaler = MinMaxScaler()
    X_df = scaler.fit_transform(X_df, Y_df)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X_df, Y_df, test_size=0.2)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size = 0.12517)

    mc_perceptron_obj = mc_perceptron.MultiClassPerceptron(visualize=visualize)

    mc_perceptron_obj.train(X_train, Y_train, 500)
    mc_perceptron_obj.save_model("./models/mc_perceptron_objv1")
    predict_val = mc_perceptron_obj.predict_class(X_val, Y_val)
    print(mc_perceptron_obj.accuracy(predict_val, Y_val))

    predict_test = mc_perceptron_obj.predict_class(X_test, Y_test)
    print(mc_perceptron_obj.accuracy(predict_test, Y_test))

def kmeans():
    X_df, Y_df = parse.preprocess()
    scaler = MinMaxScaler()
    X_df = scaler.fit_transform(X_df, Y_df)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X_df, Y_df, test_size=0.2)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size = 0.12517)

    kmeans_obj = Kmeans(Y_df.shape[1], 100)
    # kmeans_obj.train(X_data=X_df)
    # print(kmeans_obj.labels)
    # print(np.argmax(Y_df, axis=1))
    # print(kmeans_obj.compute_sse(X_df, kmeans_obj.labels, kmeans_obj.centroids))
    kmeans_obj.pca(X_df)
    # print(accuracy_score(kmeans_obj.labels, np.argmax(Y_df, axis = 1)))

    # k = KMeans(3, max_iter=100, random_state=123, n_init=10).fit(X_df)
    # print(k.score(X_df))
    
    # print(k.cluster_centers_)




if __name__ == '__main__':
    # ml_perceptron(visualize=True)
    # np.set_printoptions(threshold=sys.maxsize)
    kmeans()
    
    