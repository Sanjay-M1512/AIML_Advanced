import os
import sys
import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")


def main():
    filename = input().strip()
    file_path = os.path.join(sys.path[0], filename)

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    print("=== First 5 Rows of Data ===")
    head_df = df.head()
    print(head_df)

    print(f"The number of samples in data is {df.shape[0]}.\n")

    print("=== Data Types ===")
    print(df.dtypes)
    print()

    print("=== Statistical Summary (Describe) ===")
    desc = df.describe()
    print(desc)

    print("=== Missing Values Per Column ===")
    missing = df.isnull().sum()  
    print(missing)
    print()

    print("=== Salary Encoding Classes ===")
    le_salary = LabelEncoder()
    df["salary.enc"] = le_salary.fit_transform(df["salary"])
    salary_classes = list(le_salary.classes_)
    print(salary_classes)
    print()

    print("=== Department Encoding Classes ===")
    le_dept = LabelEncoder()
    df["Department.enc"] = le_dept.fit_transform(df["Department"])
    dept_classes = list(le_dept.classes_)
    print(dept_classes)
    print()

    print("=== Dropping 'Department' and 'salary' columns ===\n")
    df = df.drop(columns=["Department", "salary"])

    print("=== Updated DataFrame Info ===")
    df.info()


main()