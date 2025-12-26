import os
import sys
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

file_path = os.path.join(r"C:\Users\HP\OneDrive\Desktop\AIML_NOTES\dataset\ML470_S1_HR_Data_Practice.xlsx")

try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    sys.exit(1)

le_salary = LabelEncoder()
df["salary_enc"] = le_salary.fit_transform(df["salary"])

le_dept = LabelEncoder()
df["Department_enc"] = le_dept.fit_transform(df["Department"])

feature_cols = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",   # ✅ FIXED
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
    "salary_enc",
    "Department_enc"
]


features = df[feature_cols]
corr = features.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    square=True,
    linewidths=0.5,
)

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
