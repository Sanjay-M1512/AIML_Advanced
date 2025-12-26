import os
import warnings
import pandas as pd
import ML_Modules2 as mm

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")


def main():    
    file_path = os.path.join(r"C:\Users\HP\OneDrive\Desktop\AIML_NOTES\dataset\ML470_S2_Diabetes_Data_Concept.xlsx")

    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")

    cols = ["Glucose", "BMI", "Age", "FamilyHistory", "HbA1c"]
    num_df = df[cols]
    
    print("Boxplot before Outlier Treatment")
    mm.assess_outliers(num_df)

    treated_df = mm.treat_outliers(num_df)

    print("\nBoxplot after Outlier Treatment")
    mm.assess_outliers(treated_df)


if __name__ == "__main__":
    main()
