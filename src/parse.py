import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess(filename='data/data_release_17sdss.csv'): 
    df = pd.read_csv(filename, header=1)
    g_r = df['g'] - df['r']
    u_r = df['u'] - df['r']
    i_r = df['i'] - df['r']
    z_r = df['z'] - df['r']
    i_z = df['i'] - df['z']
    headers = list(df.columns)+ ['g_r', 'u_r', 'i_r', 'z_r', 'i_z']
    
    df = pd.concat([df, g_r, u_r, i_r, z_r, i_z], axis=1)
    df.set_axis(headers, axis = 1, inplace = True)
    X_df = df[['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift', 'g_r', 'u_r', 'i_r', 'z_r', 'i_z']]
    class_df = df['class']
    classes = len(class_df.unique())
    Y_df = np.zeros((class_df.shape[0], classes))
    for i in range(class_df.shape[0]):
        Y_df[i, map_class_to_integer(class_df.iloc[i])] = 1
    return X_df, Y_df

def predict(data, weights, output):
    activaiton = weights[0]
    for i in range(len(data)):
        activaiton += weights[i + 1] * data[i]
    # if activaiton
    pass

#Here we define a signum function to use later
def signum(x):
    if x > 0: return 1
    else : return -1

def train(X_train, Y_train, epochs, lr=0.01):
    X_train = X_train.to_numpy()
    weights = np.zeros((X_train.shape[1], 1))
    m = 1
    epoch = 1
    while(m != 0 and epoch <= epochs):
        for x, y in zip(X_train, Y_train):
            y_hat = signum(np.dot(weights.T, x)[0])
            # print(y_hat, y)
            if y_hat != y:
                weights = (weights.T + lr * y * x).T
                m = m + 1
        epoch += 1
       
    return weights, m

    
def map_class_to_integer(class_name):
    match class_name:
        case "STAR":
            return 0
        case "GALAXY":
            return 1
        case "QSO":
            return 2
        

def predict_class(X_predict, Y_predict, weights):
    # print(weights)
    predicted_class = np.zeros(X_predict.shape[0])
    # print(Y_predict)
    for i in range(X_predict.shape[0]):
        for j in range(Y_predict.shape[1]):
            prediction = np.dot(weights[:, j], X_predict[i,:])
            if prediction > 0:
                predicted_class[i] = j
                break
    return predicted_class

def accuracy(predicted, expected):
    error = 0
    numsamples = expected.shape[0]
    for i in range(numsamples):
        actual_class = expected[i,:]
        if actual_class[int(predicted[i])]!=1.0:
             error+=1
    return (1-error/numsamples)
    

if __name__ == '__main__':
    X_df, Y_df = preprocess()
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X_df, Y_df, test_size=0.2)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size = 0.12517)
    print("Training dataset: ", X_train.shape)
    print("Validation dataset: ", X_val.shape)
    print("Test dataset: ", X_test.shape)
    print("Training dataset: ", Y_train.shape)
    print("Validation dataset: ", Y_val.shape)
    print("Test dataset: ", Y_test.shape)
    
    weights = np.zeros((X_train.shape[1], Y_train.shape[1]))
    for i in range(Y_train.shape[1]):
        w, err = train(X_train, Y_train[:, i], 100, 0.001)
        weights[:, i] = w[:,0]

    predicted_classes = predict_class(X_val.to_numpy(), Y_val, weights)
    print(accuracy(predicted_classes, Y_val))
    predicted_classes = predict_class(X_test.to_numpy(), Y_test, weights)
    print(accuracy(predicted_classes, Y_val))
