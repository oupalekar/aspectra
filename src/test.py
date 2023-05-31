import mc_perceptron
import utils
import parse
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


## ML Perceptron ##
def ml_perceptron():
    X_df, Y_df = parse.preprocess()
    scaler = MinMaxScaler()
    X_df = scaler.fit_transform(X_df, Y_df)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X_df, Y_df, test_size=0.2)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size = 0.12517)

    mc_perceptron_obj = mc_perceptron.MultiClassPerceptron()

    mc_perceptron_obj.train(X_train, Y_train, 500)
    mc_perceptron_obj.save_model("./models/mc_perceptron_objv1")
    predict_val = mc_perceptron_obj.predict_class(X_val, Y_val)
    print(mc_perceptron_obj.accuracy(predict_val, Y_val))

    predict_test = mc_perceptron_obj.predict_class(X_test, Y_test)
    print(mc_perceptron_obj.accuracy(predict_test, Y_test))


if __name__ == '__main__':
    ml_perceptron()