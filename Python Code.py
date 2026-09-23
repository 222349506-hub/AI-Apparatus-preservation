# AI PREDICTIVE APPARATUS PRESERVATION
# ============================================================
# Project: AI Solution for Machine Industries
# Industry: Manufacturing
# Solution: AI Predictive Maintenance
# Dataset: AI4I 2020 Predictive Maintenance Dataset
# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(
    r"C:\Users\EDSON\Documents\AI-Apparatus-preservation\ai4i2020.csv"
)

# ============================================================
# 3. INITIAL DATA INSPECTION
# ============================================================

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nUnique values:")
print(df.nunique())


# ============================================================
# 4. CATEGORICAL DATA ANALYSIS
# ============================================================

print("\nCategorical variables:")

for column in df.select_dtypes(include="str").columns:
    print(f"\n{column}")
    print(df[column].value_counts(dropna=False))
# ============================================================
# 5. DATA DICTIONARY
# ============================================================
data_dictionary = pd.DataFrame({
    "Variable": [
        "UDI",
        "Product ID",
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF"
    ],
    "Role": [
        "Identifier",
        "Identifier",
        "Predictor",
        "Predictor",
        "Predictor",
        "Predictor",
        "Predictor",
        "Predictor",
        "Target",
        "Failure indicator",
        "Failure indicator",
        "Failure indicator",
        "Failure indicator",
        "Failure indicator"
    ],
    "ML_Decision": [
        "Remove",
        "Remove",
        "Keep and encode",
        "Keep",
        "Keep",
        "Keep",
        "Keep",
        "Keep",
        "Target",
        "Exclude from predictors",
        "Exclude from predictors",
        "Exclude from predictors",
        "Exclude from predictors",
        "Exclude from predictors"
    ]
})

data_dictionary

# ============================================================
# 6. TARGET VARIABLE ANALYSIS
# ============================================================

print("\nMachine failure counts:")
print(df["Machine failure"].value_counts())

print("\nMachine failure percentages:")
print(df["Machine failure"].value_counts(normalize=True) * 100)


# ============================================================
# 7. PRODUCT TYPE ANALYSIS
# ============================================================

print("\nProduct types:")
print(df["Type"].unique())

print("\nProduct type counts:")
print(df["Type"].value_counts())


# ============================================================
# 8. NUMERICAL VARIABLE ANALYSIS
# ============================================================

numerical_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

print("\nNumerical summary:")
print(df[numerical_columns].describe().T)


# ============================================================
# 9. DISTRIBUTION OF NUMERICAL VARIABLES
# ============================================================

for column in numerical_columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=column,
        kde=True
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.show()


# ============================================================
# 10. NUMERICAL VARIABLES VS MACHINE FAILURE
# ============================================================

print("\nAverage operating conditions by machine failure:")
print(
    df.groupby("Machine failure")[numerical_columns].mean()
)


# ============================================================
# 11. FAILURE RATE BY PRODUCT TYPE
# ============================================================

failure_rate = (
    df.groupby("Type")["Machine failure"]
    .mean()
    .mul(100)
)

print("\nFailure rate by product type (%):")
print(failure_rate)


# ============================================================
# 12. TOOL WEAR ANALYSIS
# ============================================================

df["Tool wear group"] = pd.cut(
    df["Tool wear [min]"],
    bins=[0, 50, 100, 150, 200, 250, 300],
    labels=[
        "0-50",
        "51-100",
        "101-150",
        "151-200",
        "201-250",
        "251-300"
    ]
)

tool_wear_failure = (
    df.groupby(
        "Tool wear group",
        observed=True
    )["Machine failure"]
    .agg(["count", "sum", "mean"])
)

tool_wear_failure["failure_rate_%"] = (
    tool_wear_failure["mean"] * 100
)

print("\nFailure analysis by tool wear group:")
print(tool_wear_failure)


# ============================================================
# 13. TORQUE ANALYSIS
# ============================================================

df["Torque group"] = pd.cut(
    df["Torque [Nm]"],
    bins=[0, 20, 30, 40, 50, 60, 70, 80],
    labels=[
        "0-20",
        "21-30",
        "31-40",
        "41-50",
        "51-60",
        "61-70",
        "71-80"
    ]
)

torque_failure = (
    df.groupby(
        "Torque group",
        observed=True
    )["Machine failure"]
    .agg(["count", "sum", "mean"])
)

torque_failure["failure_rate_%"] = (
    torque_failure["mean"] * 100
)

print("\nFailure analysis by torque group:")
print(torque_failure)


# ============================================================
# 14. TORQUE AND TOOL WEAR ANALYSIS
# ============================================================

torque_toolwear = pd.crosstab(
    df["Tool wear group"],
    df["Torque group"],
    values=df["Machine failure"],
    aggfunc="mean"
) * 100

print("\nFailure percentage by tool wear and torque:")
print(torque_toolwear)


# ============================================================
# 15. FAILURE MODE ANALYSIS
# ============================================================

failure_modes = [
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]

print("\nFailure mode counts:")
print(
    df[df["Machine failure"] == 1][failure_modes].sum()
)

failure_mode_rates = (
    df[df["Machine failure"] == 1][failure_modes]
    .mean()
    .mul(100)
)

print("\nFailure mode percentages:")
print(failure_mode_rates)


# ============================================================
# 16. PREPARE DATA FOR MACHINE LEARNING
# ============================================================

model_columns = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure"
]

model_df = df[model_columns].copy()

print("\nModel dataset:")
print(model_df.head())


# ============================================================
# 17. SEPARATE FEATURES AND TARGET
# ============================================================

X = model_df.drop(
    "Machine failure",
    axis=1
)

y = model_df["Machine failure"]

print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("\nTarget counts:")
print(y.value_counts())

print("\nTarget percentages:")
print(
    y.value_counts(normalize=True) * 100
)


# ============================================================
# 18. ENCODE CATEGORICAL VARIABLE
# ============================================================

X_encoded = pd.get_dummies(
    X,
    columns=["Type"],
    dtype=int
)

print("\nEncoded feature shape:")
print(X_encoded.shape)

print("\nEncoded feature data types:")
print(X_encoded.dtypes)

print("\nFirst five encoded records:")
print(X_encoded.head())


# ============================================================
# 19. MACHINE LEARNING MODEL
# ============================================================
# 
#
# Planned steps:
# 1. Split data into training and testing sets
# 2. Train the predictive maintenance model
# 3. Make predictions
# 4. Evaluate the model
# 5. Display the confusion matrix
# ============================================================