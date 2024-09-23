import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

# 1. Load MNIST Data
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1)
    Xraw = mnist.data / 255.0  # Normalize pixel values between 0 and 1
    Yraw = mnist.target.astype(int)
    return Xraw, Yraw

Xraw, Yraw = load_mnist()

# Plot one example of each digit
def plot_digits(Xraw, Yraw):
    plt.figure(figsize=(10, 4))
    for i in range(10):
        ax = plt.subplot(2, 5, i + 1)
        digit_image = Xraw[Yraw == i][0].reshape(28, 28)
        ax.imshow(digit_image, cmap='gray')
        plt.title(f'Digit {i}')
        plt.axis('off')
    plt.show()

plot_digits(Xraw, Yraw)

# 2. Pre-process data
def preprocess_data(Xraw, Yraw, d):
    # Create M matrix
    M = np.random.uniform(0, 1, (d, 784)) / (255 * d)
    
    # Create X matrix (d x 70,000)
    X = np.dot(M, Xraw.T)
    
    # One-hot encode Y
    encoder = OneHotEncoder(sparse=False, categories='auto')
    Y = encoder.fit_transform(Yraw.reshape(-1, 1)).T
    
    return X, Y

# 3. Moore-Penrose pseudoinverse to find W
def linear_regression(X, Y):
    W = np.dot(Y, np.linalg.pinv(X))
    return W

# 3. Calculate MSE and number of mistakes
def evaluate_model(X, Y, W):
    Y_pred = np.dot(W, X)
    MSE = np.mean((Y - Y_pred) ** 2)
    
    Y_true_labels = np.argmax(Y, axis=0)
    Y_pred_labels = np.argmax(Y_pred, axis=0)
    
    mistakes = np.sum(Y_true_labels != Y_pred_labels)
    return MSE, mistakes

# Test for various d values
d_values = [10, 50, 100, 200, 500]
for d in d_values:
    X, Y = preprocess_data(Xraw, Yraw, d)
    W = linear_regression(X, Y)
    mse, mistakes = evaluate_model(X, Y, W)
    print(f"d = {d}, MSE = {mse}, Mistakes = {mistakes}")

# 4. Widrow-Hoff LMS Algorithm
def widrow_hoff(X, Y, d, eta=0.001, epochs=10):
    W = np.zeros((Y.shape[0], X.shape[0]))  # Initialize W at zero
    MSEs = []
    
    for epoch in range(epochs):
        for i in range(X.shape[1]):  # Update weights for each sample
            error = Y[:, i] - np.dot(W, X[:, i])
            W += eta * np.outer(error, X[:, i])
        
        # Calculate MSE after each epoch
        mse = np.mean((Y - np.dot(W, X)) ** 2)
        MSEs.append(mse)
    
    return W, MSEs

# Running Widrow-Hoff LMS for d = 100
d = 100
X, Y = preprocess_data(Xraw, Yraw, d)
W_lms, MSEs_lms = widrow_hoff(X, Y, d)

# Plot the MSE vs epochs for LMS
plt.figure()
plt.plot(range(1, 11), MSEs_lms, marker='o')
plt.title("MSE vs Epochs for Widrow-Hoff LMS (d=100)")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

# Evaluate LMS weights
mse_lms, mistakes_lms = evaluate_model(X, Y, W_lms)
print(f"LMS MSE: {mse_lms}, LMS Mistakes: {mistakes_lms}")

# 5. Dimensionality Reduction and Plot
def plot_pca_images(Xraw, Yraw, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(Xraw)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Yraw, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f'MNIST PCA with {n_components} Components')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.show()

plot_pca_images(Xraw, Yraw, n_components=2)
