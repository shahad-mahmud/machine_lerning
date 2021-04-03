from os import error
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

data_file = datasets.load_boston()  # get the boston house-prices data set

# have a look at the attributes
print(data_file.keys())

# get the data
x = data_file.data
x = np.array(x)  # convert to numpy array
print(x.shape)

# the data has a shape of (506, 13)
# we need a neumeric value. So average is taken
x = np.average(x, axis=1)
print(x)

# plt.figure()
# plt.scatter(range(len(x)), x)
# plt.show()


# split data into train and test
X_train = x[:450]
X_test = x[450:]
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

Y_train = data_file.target[:450]
Y_train = np.array(Y_train)
Y_test = data_file.target[450:]


a_0 = np.zeros((450, 1))
a_1 = np.zeros((450, 1))

learning_rate = 0.0001
epoch = 0

while epoch < 500:
    y = a_0 + a_1 * X_train
    error = y - Y_train
    mse = np.sum(error ** 2)
    mse =mse / 450
    a_0 = a_0 - learning_rate * 2 * np.sum(error) / 450
    a_1 = a_1 - learning_rate * 2 * np.sum(error * X_train) / 450
    epoch += 1
    if epoch % 25 == 0:
        print('Epoch {}: error {}'.format(epoch, mse))