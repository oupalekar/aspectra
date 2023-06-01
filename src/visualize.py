import matplotlib.pyplot as plt
import time


def error_vs_epochs(error, epoch, filename):
    plt.plot(epoch, error)
    plt.ylabel("Error percentage")
    plt.xlabel("Number of Epochs")
    plt.title("Error vs Epochs")

    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("(%m-%d-%Y-%H:%M:%S)", named_tuple)
    plt.savefig(f'results/epoch_vs_error{time_string}')