import numpy as np
from sklearn.model_selection import train_test_split


class LDA:

    def __init__(self):
        # Initialize instance variables
        self.mean_vectors = None
        self.shared_covariance = None
        self.w = None
        self.b = None
        self.classes = None

    def fit(self, X, y, val_size=0.1):
        # Split data into training and validation sets
        self.classes = np.unique(y)
        train_X, val_X, train_y, val_y = train_test_split(
            X, y, test_size=val_size, random_state=42)

        # Compute the mean vector for each class
        self.mean_vectors = np.array(
            [np.mean(train_X[train_y == c], axis=0) for c in self.classes])

        # Compute the shared covariance matrix
        n = train_X.shape[0]
        # Calculate covariance matrix for each class
        covariance_matrices = [np.cov(train_X[train_y == c].T)
                               for c in self.classes]
        class_sizes = [train_X[train_y == c].shape[0]
                       for c in self.classes]  # Get number of samples in each class
        # Compute sum of weighted covariance matrices
        sigma_sum = sum([class_sizes[i]*covariance_matrices[i]
                        for i in range(len(self.classes))])
        # Calculate the shared covariance matrix
        self.shared_covariance = sigma_sum / n

        # Compute the weight and bias for each class
        self.w = []
        self.b = []
        for i, c in enumerate(self.classes):
            sigma_inv = np.linalg.inv(self.shared_covariance)
            mean = self.mean_vectors[i]
            w = sigma_inv.dot(mean)  # Compute weight
            b = -0.5 * mean.dot(sigma_inv).dot(mean) + np.log(class_sizes[i]/n) + np.log(
                self.compute_likelihood_ratio(mean))  # Compute bias
            self.w.append(w)
            self.b.append(b)
        self.w = np.array(self.w).reshape(len(self.classes), -1)
        self.b = np.array(self.b).reshape(-1)

        # Compute the accuracy score on the validation set
        y_pred = self.predict(val_X)
        accuracy = np.mean(y_pred == val_y)
        print(f"Validation accuracy: {accuracy:.2f}")

    def compute_likelihood_ratio(self, mean_vector):
        # Compute the likelihood ratio for a given class
        diff = mean_vector - self.mean_vectors
        sigma_inv = np.linalg.inv(self.shared_covariance)
        numerator = np.exp(-0.5 * diff.dot(sigma_inv).dot(diff.T))
        denominator = np.sqrt(np.linalg.det(self.shared_covariance))
        return numerator / denominator

    def predict(self, X):
        # Make predictions on a given set of samples
        y_pred = []
        for sample in X:
            # Compute the discriminant function for each class
            discriminant_functions = [self.w[k].T.dot(
                sample) + self.b[k] for k in range(len(self.classes))]

            # Compute the class probabilities using softmax function
            class_probabilities = self.softmax(discriminant_functions)

            # Assign the sample to the class with the highest probability
            predicted_class = self.classes[np.argmax(class_probabilities)]
            y_pred.append(predicted_class)
        return np.array(y_pred)

    def softmax(self, z):
        # Compute the softmax function
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)
