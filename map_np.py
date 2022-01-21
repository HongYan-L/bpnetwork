import random
import numpy as np
import mnist_loader

def sigmoid(z):

    return 1.0 / (1.0 + np.exp(-z))

def dsigmoid(z):

    return sigmoid(z)(1 - sigmoid(z))


class MLP_np:

    def __init__(self, sizes):
        """

        :param sizes: [784, 30, 10]
        """
        self.sizes = sizes
        self.num_layers = len(sizes) - 1

        # sizes:[784, 30, 10]
        # w:[ch_out, ch_in]
        # b:[ch_out]
        self.weights = [np.random.randn(ch2, ch1) for ch1, ch2 in zip(sizes[:-1], sizes[1:])] # [784, 30], [30, 10]
        self.biases = [np.random.randn(ch, 1) for ch in sizes[1:]]

    def forward(self, a):
        """

        :param x:[784, 1]
        :return:[10, 1]
        """
        for b, w in zip(self.biases, self.weights):
            # [30, 784]@[784, 1] => [30, 1] + [30, 1] => [30, 1]
            a = sigmoid(np.dot(w, a) + b)

        return a


    def backward(self, a, y):
        """

        :param a:[784, 1]
        :param y:[10, 1], one_hot encoding
        :return:
        """

        # save activation for every layer
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        activations = [a]
        # save z for every layer
        z_tmp = []
        activation = a

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            activation = sigmoid(z)
            z_tmp.append(z)
            activations.append(activation)

        loss = np.power(activations[-1] - y, 2).sum()
        # compute gradient on output layer
        # [10, 1]
        delta = activations[-1] * (1 - activations[-1]) * (activations[-1] - y)
        nabla_b[-1] = delta
        # [10, 1] @ [1, 30] => [10, 30]
        # activation:[30, 1]
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # compute hidden gradient
        for l in range(2, self.num_layers + 1):
            l = -l

            z = z_tmp[l]
            x = activations[l]

            delta = np.dot(self.weights[l+1].T, delta) * x * (1 - x)

            nabla_b[l] = delta
            # [30, 1] @ [784, 1]T => [30, 784]
            nabla_w[l] = np.dot(delta, activations[l - 1].T)

        return nabla_w, nabla_b, loss


    def train(self, training_data, epochs, batchsz, lr, test_data):
        """

        :param training_data: list of (x,y)
        :param epochs: 1000
        :param batchsz: 10
        :param lr: 0.01
        :param test_data: list of (x,y)
        :return:
        """
        if test_data:
            n_test = len(test_data)

        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+batchsz] for k in range(0, n, batchsz)]

            for mini_batch in mini_batches:
                loss = self.update_mini_batch(mini_batch, lr)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test), loss)
            else:
                print("Epoch {0} complete".format(j))


    def update_mini_batch(self, batch, lr):
        """

        :param self:
        :param batch: list of (x,y)
        :param lr: 0.01
        :return:
        """
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        loss = 0

        # for every sample in current batch
        for x, y in batch:
            nabla_w_, nabla_b_, loss_ = self.backward(x, y)
            nabla_w = [accu + cur for accu, cur in zip(nabla_w, nabla_w_)]
            nabla_b = [accu + cur for accu, cur in zip(nabla_b, nabla_b_)]
            loss += loss_

        nabla_w = [w/len(batch) for w in nabla_w]
        nabla_b = [b/len(batch) for b in nabla_b]
        loss = loss / len(batch)

        # w = w - lr * nabla_w
        self.weights = [w - lr * nabla for w, nabla in zip(self.weights, nabla_w)]
        self.biases = [b - lr * nabla for b, nabla in zip(self.biases, nabla_b)]

        return loss


    def evaluate(self, test_data):
        """

        :param test_data: list of (x,y)
        :return:
        """
        result = [(np.argmax(self.forward(x)), y) for x, y in test_data]

        correct = sum(int(pred == y) for pred, y in result)

        return correct


def main():

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    # set up a network with 30 hidden neurons
    net = MLP_np([784, 30, 10])
    net.train(training_data, 1000, 10, 0.1, test_data=test_data)


if __name__ == '__main__':

    main()
