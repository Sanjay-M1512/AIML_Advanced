import numpy as np
import matplotlib.pyplot as plt


def plot_svm_fruit_classification(X, y, model):
    """
    Visualize linear SVM decision boundary, margins, and support vectors
    for a 2D fruit dataset.
    X: numpy array of shape (n_samples, 2)
    y: labels (0 = apples, 1 = oranges)
    model: trained linear SVM (e.g., SVC(kernel='linear'))
    """

    plt.figure(figsize=(7, 5))

    # 1. Plot apples and oranges
    apples = y == 0
    oranges = y == 1
    plt.scatter(
        X[apples, 0],
        X[apples, 1],
        c="green",
        label="Apples",
        edgecolors="k",
    )
    plt.scatter(
        X[oranges, 0],
        X[oranges, 1],
        c="orange",
        label="Oranges",
        edgecolors="k",
    )

    # 2. Get current axis limits to draw lines nicely
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create a grid to evaluate the decision function
    xx = np.linspace(xlim[0], xlim[1], 200)
    # For linear SVM, decision function is w.x + b = 0
    w = model.coef_[0]
    b = model.intercept_[0]

    # Decision boundary: w0 * x + w1 * y + b = 0  →  y = -(w0/w1)x - b/w1
    yy = -(w[0] / w[1]) * xx - b / w[1]

    # Margins: w0 * x + w1 * y + b = ±1  →  y = -(w0/w1)x - (b ± 1)/w1
    margin_up = -(w[0] / w[1]) * xx - (b + 1) / w[1]
    margin_down = -(w[0] / w[1]) * xx - (b - 1) / w[1]

    # 3. Plot decision boundary and margins
    plt.plot(xx, yy, "k-", label="Decision boundary")
    plt.plot(xx, margin_up, "k--")
    plt.plot(xx, margin_down, "k--")

    # 4. Plot support vectors (no fill, clear outline)
    sv = model.support_vectors_
    plt.scatter(
        sv[:, 0],
        sv[:, 1],
        s=120,
        facecolors="none",
        edgecolors="k",
        linewidths=1.5,
        label="Support Vectors",
    )

    plt.xlabel("Size of Fruit")
    plt.ylabel("Sweetness Level")
    plt.title("SVM Decision Boundary for Fruit Classification")
    plt.legend()
    plt.tight_layout()
    plt.show()