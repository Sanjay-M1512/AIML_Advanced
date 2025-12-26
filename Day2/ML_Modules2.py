import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def assess_outliers(data):
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=data.select_dtypes(include=["number"]))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def treat_outliers(data):
    df = data.copy().astype(float)

    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    for col in df.columns:
        df[col] = np.where(df[col] < lower[col], lower[col], df[col])
        df[col] = np.where(df[col] > upper[col], upper[col], df[col])

    return df