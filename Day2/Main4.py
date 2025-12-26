import warnings
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import ML_Modules4 as mm

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")


def main():
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        n_classes=2,
        random_state=42,
    )

    model = SVC(kernel="linear", C=1.0)
    model.fit(X, y)

    mm.plot_svm_fruit_classification(X, y, model)


if __name__ == "__main__":
    main()