# log-reg
A logistic regression library that does multi-class log-reg instead of the very basic binary log-reg

## Techniques Used
- Logistic Regression
- Softmax Activation
- Gradient Descent
- One-Hot Encoding
- Sigmoid Function

## Functionality

#### Log Reg Class
It includes methods for softmax activation, one-hot encoding, and fitting the logistic regression model using gradient descent.
- ##### Softmax Activation Method
    The computes the softmax activation which is used to convert raw scores (logits) into class probabilities for multiclass classification for a given input vector, which is used for multiclass classification to convert raw scores into class probabilities.
- ##### One Hot Encode Method
    It converts target labels into a one-hot encoded format so that each class is represented as a binary vector, where the index corresponding to the true class is set to 1, and all other indices are set to 0, this is essential for multiclass log-reg.
- ##### Predict Method
  It uses the trained weights and bias to predict class labels for input data, applying softmax activation and selects the class with the highest probability as the prediction.
- ##### Fit Method
    It trains the logistic regression model using the provided input features (X) and target labels (y) using batch gradient descent optimization.
    The weights and biases are updated iteratively to minimize the error between the predicted probabilities and the one-hot encoded target labels.
