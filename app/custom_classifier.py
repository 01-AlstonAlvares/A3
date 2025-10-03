import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    """
    Custom implementation of Multinomial Logistic Regression using Gradient Descent.
    Supports batch, stochastic, and mini-batch gradient descent.
    """
    def __init__(self, lr=0.1, num_iter=10000, reg_type='l2', lambda_=0.1, 
                 verbose=False, gd_type='batch', batch_size=32):
        """Initializes the LogisticRegression classifier."""
        self.initial_lr = lr # Store the initial learning rate
        self.num_iter = num_iter
        self.reg_type = reg_type
        self.lambda_ = lambda_
        self.verbose = verbose
        self.gd_type = gd_type
        self.batch_size = batch_size
        self.weights = None
        self.loss_history = []

    def _add_intercept(self, X):
        return np.insert(X, 0, 1, axis=1)

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot(self, y, n_classes):
        return np.eye(n_classes)[y]

    def _compute_loss(self, X_bias, y_one_hot):
        n_samples = X_bias.shape[0]
        scores = X_bias.dot(self.weights)
        probs = self._softmax(scores)
        
        core_loss = -np.sum(y_one_hot * np.log(probs + 1e-9)) / n_samples
        if self.reg_type == 'l2':
            l2_penalty = (self.lambda_ / (2 * n_samples)) * np.sum(np.square(self.weights[1:, :]))
            return core_loss + l2_penalty
        return core_loss

    def fit(self, X, y):
        """
        Trains the logistic regression model using the specified gradient descent type.
        """
        X_bias = self._add_intercept(X)
        n_samples, n_features = X.shape
        self.n_classes_ = len(np.unique(y))
        self.loss_history = []

        self.weights = np.random.randn(n_features + 1, self.n_classes_) * 0.01
        y_one_hot = self._one_hot(y, self.n_classes_)
        
        # --- FIX: Use a much smaller decay rate for a more gradual schedule ---
        decay_rate = 0.01 

        # Main training loop
        for i in range(self.num_iter):
            # --- UPDATE: Implement a learning rate schedule ---
            # The learning rate will decrease over time for more stable convergence.
            lr = self.initial_lr / (1 + decay_rate * i)

            if self.gd_type == 'batch':
                indices = np.arange(n_samples)
            elif self.gd_type == 'stochastic':
                indices = np.random.randint(0, n_samples, 1)
            elif self.gd_type == 'mini':
                indices = np.random.choice(n_samples, self.batch_size, replace=False)
            else:
                raise ValueError("gd_type must be one of 'batch', 'stochastic', or 'mini'")
            
            X_batch = X_bias[indices]
            y_batch = y_one_hot[indices]

            scores = X_batch.dot(self.weights)
            probabilities = self._softmax(scores)
            error = probabilities - y_batch
            gradient = (1 / len(X_batch)) * X_batch.T.dot(error)

            if self.reg_type == 'l2':
                gradient[1:, :] += (self.lambda_ / len(X_batch)) * self.weights[1:, :]

            # Update weights using the current, decayed learning rate
            self.weights -= lr * gradient

            # Record loss at regular intervals
            if i % 100 == 0: 
                loss = self._compute_loss(X_bias, y_one_hot)
                self.loss_history.append(loss)
                if self.verbose and i % 1000 == 0:
                    print(f"Iteration {i}, Loss: {loss:.4f}, LR: {lr:.6f}")

    def predict_proba(self, X):
        X_bias = self._add_intercept(X)
        scores = X_bias.dot(self.weights)
        return self._softmax(scores)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

