import numpy as np
import matplotlib.pyplot as plt


def plot_moons_dataset(X, y):
    """
    Basic scatter plot of the raw Moons dataset.
    X: array of shape (n_samples, 2)
    y: class labels (0, 1)
    """
    plt.figure(figsize=(6, 5))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="tab:blue", edgecolors="k", label="Class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="tab:orange", edgecolors="k", label="Class 1")
    plt.title("Moons Dataset (Raw)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_svm_decision_boundary(model, X, y, title):
    """
    Plot decision regions of an SVM model and overlay data points.
    Works for both linear and RBF SVM with 2D input.
    """
    plt.figure(figsize=(6, 5))

    # Define grid over feature space
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )

    # Predict over grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Plot decision regions
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

    # Overlay training points
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="tab:blue", edgecolors="k", label="Class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="tab:orange", edgecolors="k", label="Class 1")

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.tight_layout()
    plt.show()