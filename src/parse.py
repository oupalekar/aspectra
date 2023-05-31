import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import map_class_to_integer

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

if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    X_df, Y_df = preprocess()
    scaler = MinMaxScaler()
    X_df = scaler.fit_transform(X_df, Y_df)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X_df, Y_df, test_size=0.2)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size = 0.12517)
    print("Training dataset: ", X_train.shape)
    print("Validation dataset: ", X_val.shape)
    print("Test dataset: ", X_test.shape)
    print("Training dataset: ", Y_train.shape)
    print("Validation dataset: ", Y_val.shape)
    print("Test dataset: ", Y_test.shape)

