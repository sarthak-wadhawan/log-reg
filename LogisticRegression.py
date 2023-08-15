import numpy as np


class LogisticRegression:
    def __init__(self, alpha=0.01, max_epochs=1000, num_classes=3):
        self.alpha = alpha
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None
        self.num_classes = num_classes

    def softmax(self, vector):
        e = np.exp(vector)
        return (e / np.sum(e, axis=1, keepdims=True))

    # def sigmoid(self, z):
    #     return 1/(1 + np.exp(-z))

    def _one_hot_encode(self, y, num_classes):
        encoded_y = np.zeros((len(y), num_classes))
        for i in range(len(y)):
            encoded_y[i][y[i]] = 1
        return encoded_y

    def fit(self, X, y):
        # add bias term to X
        X = np.c_[np.ones(X.shape[0]), X]
        self.num_classes = len(np.unique(y))

        # initialize weights and bias to zero
        self.weights = np.zeros((X.shape[1], self.num_classes))
        self.bias = np.zeros(self.num_classes)

        # I'm using gradient descent to optimize parameters
        for i in range(self.max_epochs):
            y_linear = np.dot(X, self.weights) + self.bias
            y_predicted = self.softmax(y_linear)
            error = y_predicted - self._one_hot_encode(y, self.num_classes)

            dw = np.dot(X.T, error) / len(y)
            db = np.sum(error, axis=0) / len(y)
            # update weights and bias -> w = w - alpha*dw, b = b- alpha*db
            self.weights = self.weights - self.alpha * dw
            self.bias = self.bias - self.alpha * db
        # print("Weights: ", self.weights.shape)
        # print("Bias: ", self.bias.shape)
        # print("X", X.shape)

    def predict(self, X):
        # add bias term to X
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        # print("X:", X.shape)
        # predict class labels
        # print("X", X.shape)
        # print(" Weights: ", self.weights.shape)
        # print("Bias: ", self.bias.shape)
        y_linear = np.dot(X, self.weights) + self.bias
        y_pred = self.softmax(y_linear)
        class_pred = np.argmax(y_pred, axis=1)
        return class_pred
