import warnings
from sklearn.datasets import make_moons
from sklearn.svm import SVC
import ML_Modules5 as mm

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")


def main():
    # Generate Moons dataset
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

    # 1. Visualization of raw dataset
    mm.plot_moons_dataset(X, y)

    # 2. Linear SVM
    linear_svm = SVC(kernel="linear", C=1.0)
    linear_svm.fit(X, y)
    mm.plot_svm_decision_boundary(linear_svm, X, y, "Linear SVM Decision Boundary on Moons")

    # 3. Non-linear SVM (RBF kernel)
    rbf_svm = SVC(kernel="rbf", C=1.0, gamma="scale")
    rbf_svm.fit(X, y)
    mm.plot_svm_decision_boundary(rbf_svm, X, y, "RBF SVM Decision Boundary on Moons")


if __name__ == "__main__":
    main()