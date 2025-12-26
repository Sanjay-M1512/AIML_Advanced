import os
import sys
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

file_path = r"C:\Users\HP\OneDrive\Desktop\AIML_NOTES\dataset\ML470_S1_HR_Data_Practice.xlsx"

try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    sys.exit(1)

numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(10, 4))
sns.boxplot(data=numeric_df, color="red")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
