import numpy as np
import mlflow

class LogisticRegression:
    """
    A custom multi-class Logistic Regression model with L2 regularization
    and MLflow logging capabilities for per-epoch metrics.
    """
    def __init__(self, lr=0.01, num_iter=2000, reg_type='l2', lambda_=0.1, log_every=100, verbose=False):
        """
        Initializes the model.
        
        Args:
            lr (float): The learning rate for gradient descent.
            num_iter (int): The number of training iterations (epochs).
            reg_type (str): The type of regularization ('l2' or 'none').
            lambda_ (float): The regularization strength.
            log_every (int): The interval (in iterations) for logging metrics to MLflow.
            verbose (bool): If True, prints progress during training.
        """
        self.lr = lr
        self.num_iter = num_iter 
        self.reg_type = reg_type
        self.lambda_ = lambda_
        self.verbose = verbose
        self.log_every = log_every
        self.weights = None
        
    def _softmax(self, z):
        """Computes the softmax function for numerical stability."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot(self, y, num_classes):
        """Converts a vector of labels to a one-hot encoded matrix."""
        return np.eye(num_classes)[y]

    def fit(self, X, y, eval_set=None):
        """
        Trains the model on the given data and logs metrics over epochs.
        
        Args:
            X (np.array): The training feature data.
            y (np.array): The training target labels.
            eval_set (tuple, optional): A tuple (X_val, y_val) for validation and logging.
                                        If provided, validation accuracy is logged every 'log_every' iterations.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Add bias term to feature matrix
        X_bias = np.insert(X, 0, 1.0, axis=1)
        n_samples, n_features = X_bias.shape
        self.num_classes = len(np.unique(y))
        
        # Initialize weights to zeros
        self.weights = np.zeros((n_features, self.num_classes))
        y_one_hot = self._one_hot(y, self.num_classes)
        
        # Unpack validation data if provided for logging
        if eval_set:
            X_val, y_val = eval_set
        
        # --- Training Loop (Epochs) ---
        for iteration in range(self.num_iter):
            logits = X_bias.dot(self.weights)
            probabilities = self._softmax(logits)
            error = probabilities - y_one_hot
            
            # Calculate gradient of the cross-entropy loss
            grad = (1 / n_samples) * X_bias.T.dot(error)
            
            # Add L2 regularization term (Ridge) to the gradient
            if self.reg_type == 'l2':
                reg_term = (self.lambda_ / n_samples) * self.weights
                reg_term[0, :] = 0  # Do not regularize the bias term
                grad += reg_term
                
            # Update weights using gradient descent
            self.weights -= self.lr * grad
            
            # --- Per-Epoch Logging to MLflow ---
            if (iteration % self.log_every == 0) and eval_set:
                accuracy = self.score(X_val, y_val)
                # Use 'step' to create a time-series plot in MLflow
                mlflow.log_metric("validation_accuracy", accuracy, step=iteration)

            if self.verbose and iteration % 500 == 0:
                print(f"Iteration {iteration}/{self.num_iter}")
                
    def predict(self, X):
        """Makes class predictions for the input data."""
        X = np.asarray(X)
        X_bias = np.insert(X, 0, 1.0, axis=1)
        logits = X_bias.dot(self.weights)
        probabilities = self._softmax(logits)
        return np.argmax(probabilities, axis=1)

    def score(self, X, y):
        """Calculates the accuracy of the model on the given data."""
        y_pred = self.predict(X)
        return np.mean(y_pred == np.asarray(y))