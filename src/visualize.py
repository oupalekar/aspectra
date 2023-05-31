import matplotlib.pyplot as plt


def error_vs_epochs(error, epoch, filename):
    plt.plot(epoch, error)
    plt.ylabel("Error percentage")
    plt.xlabel("Number of Epochs")
    plt.title("Error vs Epochs")
    plt.savefig('results/epoch_vs_errorV1')