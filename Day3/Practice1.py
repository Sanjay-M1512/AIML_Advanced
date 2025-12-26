import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    
    filename=input()
    filepath=os.path.join(sys.path[0],filename)
    
    try:
        df= pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    
    print("First 5 rows of the dataset:")
    print(df.head())
    print()
    
    print("Number of samples in the data:")
    print(len(df))
    print()
    
    print("Data types of each column:")
    print(df.dtypes)
    print()
    
    print("Feature columns:")
    feature_cols = [
        'satisfaction_level', 'last_evaluation', 'number_project',
        'average_montly_hours', 'time_spend_company',
        'Work_accident', 'promotion_last_5years'
    ]
    target_col = 'left'
    print(feature_cols)
    print()
    
    print("Statistical summary of numeric columns:")
    print(df.describe())
    print()

    X= df[feature_cols].values
    y= df[target_col].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", accuracy)
    print()
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()