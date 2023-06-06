import sys
import perceptron
import utils
import parse
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from kmeans import *
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans, DBSCAN


## ML Perceptron ##
def ml_perceptron(visualize = False):
    X_df, Y_df = parse.preprocess()
    scaler = MinMaxScaler()
    X_df = scaler.fit_transform(X_df, Y_df)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X_df, Y_df, test_size=0.2)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size = 0.12517)

    mc_perceptron_obj = perceptron.MultiClassPerceptron(visualize=visualize)

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
    print(X_df.shape)
    # pca_data_set = Kmeans(3, max_iters=100).pca(X_df).T @ X_df.T

    kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    # print(pca_data_set.shape)
    kmeans.fit(X_df)
    labels = kmeans.labels_
    print(np.argmax(Y_df, axis = 1), labels)
    print(sum(np.argmax(Y_df, axis = 1) == labels)/len(labels))

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # for i in range(Y_df.shape[0]):
    #     if labs[i] == 0:
    #         color_val = 'red'
    #     elif labs[i] == 1:
    #         color_val = 'blue'
    #     else:
    #         color_val = 'green'
    #     ax.scatter(pca_data_set[0,i], pca_data_set[1,i], pca_data_set[2,i], marker='o', color = color_val, alpha=0.1)
    # ax.set_xlabel('$x$', rotation=150)
    # ax.set_ylabel('$y$')
    # ax.set_zlabel('$z$',rotation=60)
    # # ax.scatter(kmeans_obj.centroids[:,0], kmeans_obj.centroids[:,1], kmeans_obj.centroids[:,2], marker ='o', color="black")
    # plt.savefig(f'results/kmeans_pca_scatter')
    # plt.show()

def dbscan():
    X_df, Y_df = parse.preprocess()
    scaler = MinMaxScaler()
    X_df = scaler.fit_transform(X_df, Y_df)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X_df, Y_df, test_size=0.2)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size = 0.12517)

    # print(X_df.shape)

    kmeans_obj = Kmeans(Y_df.shape[1], 100, 64)
    # kmeans_obj.train(X_df, Y_df)

    # print(kmeans_obj.compute_sse(X_df, kmeans_obj.labels, kmeans_obj.centroids))
    # print(accuracy_score(kmeans_obj.labels, np.argmax(Y_df, axis = 1)))
    pca_vec = kmeans_obj.pca(X_df)
    pca_data_set = pca_vec.T @ X_df.T

    db_obj = DBSCAN(10).fit(pca_data_set.T.real, Y_df)
    print(db_obj.labels_)
    




if __name__ == '__main__':
    # ml_perceptron(visualize=False)
    # np.set_printoptions(threshold=sys.maxsize)
    kmeans()

    
    