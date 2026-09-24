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
# 19. MANUAL TEST SPLIT
# ============================================================
np.random.seed(42)

failure_indices = y[y == 1].index

non_failure_indices = y[y == 0].index


# Shuffle each class

failure_indices = np.random.permutation(
    failure_indices
)

non_failure_indices = np.random.permutation(
    non_failure_indices
)


# Use 80% of each class for training

failure_train_size = int(
    len(failure_indices) * 0.8
)

non_failure_train_size = int(
    len(non_failure_indices) * 0.8
)


failure_train = (
    failure_indices[:failure_train_size]
)

failure_test = (
    failure_indices[failure_train_size:]
)


non_failure_train = (
    non_failure_indices[:non_failure_train_size]
)

non_failure_test = (
    non_failure_indices[non_failure_train_size:]
)


# Combine the two classes

train_indices = np.concatenate([
    failure_train,
    non_failure_train
])

test_indices = np.concatenate([
    failure_test,
    non_failure_test
])


# Shuffle final sets

np.random.shuffle(train_indices)

np.random.shuffle(test_indices)


# Create training and testing datasets

X_train = X_encoded.loc[
    train_indices
]

X_test = X_encoded.loc[
    test_indices
]

y_train = y.loc[
    train_indices
]

y_test = y.loc[
    test_indices
]


print("\nTraining/testing shapes:")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


print("\nTraining set:")
print(y_train.value_counts())

print(
    y_train.value_counts(
        normalize=True
    ) * 100
)


print("\nTesting set:")
print(y_test.value_counts())

print(
    y_test.value_counts(
        normalize=True
    ) * 100
)


# ============================================================
# 20. SCALE NUMERICAL FEATURES
# ============================================================

scaler = StandardScaler()

numeric_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


X_train_scaled = X_train.copy()

X_test_scaled = X_test.copy()


X_train_scaled[numeric_features] = (
    scaler.fit_transform(
        X_train[numeric_features]
    )
)


X_test_scaled[numeric_features] = (
    scaler.transform(
        X_test[numeric_features]
    )
)


print("\nScaled training data summary:")

print(
    X_train_scaled[
        numeric_features
    ].describe()
)


# ============================================================
# 21. LOGISTIC REGRESSION
# ============================================================

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

print(
    "\nLogistic Regression model trained successfully."
)


# Predictions

y_pred = logistic_model.predict(
    X_test_scaled
)


print("\nFirst 20 Logistic Regression predictions:")
print(y_pred[:20])


print("\nFirst 20 actual values:")
print(y_test[:20].values)


# ============================================================
# 22. LOGISTIC REGRESSION EVALUATION
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nLogistic Regression Confusion Matrix:")
print(cm)


print(
    "\nAccuracy:",
    accuracy_score(y_test, y_pred)
)

print(
    "Precision:",
    precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
)

print(
    "F1-score:",
    f1_score(
        y_test,
        y_pred,
        zero_division=0
    )
)


print(
    "\nLogistic Regression Classification Report:"
)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ROC-AUC

y_prob = (
    logistic_model
    .predict_proba(X_test_scaled)[:, 1]
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print(
    "Logistic Regression ROC-AUC:",
    roc_auc
)


# ============================================================
# 23. LOGISTIC REGRESSION FAILURE PROBABILITY
# ============================================================

failure_probabilities = (
    y_prob[y_test == 1]
)

print(
    "\nNumber of actual failures:",
    len(failure_probabilities)
)

print(
    "\nFailure probability summary:"
)

print(
    pd.Series(
        failure_probabilities
    ).describe()
)


print(
    "\nActual failures with probability >= 0.5:",
    (
        failure_probabilities >= 0.5
    ).sum()
)

print(
    "Actual failures with probability >= 0.3:",
    (
        failure_probabilities >= 0.3
    ).sum()
)

print(
    "Actual failures with probability >= 0.2:",
    (
        failure_probabilities >= 0.2
    ).sum()
)


# ============================================================
# 24. LOGISTIC REGRESSION THRESHOLD ANALYSIS
# ============================================================

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]


print("\nLogistic Regression threshold analysis:")

print(
    "Threshold | Precision | Recall | F1-score"
)

print("-" * 45)


for threshold in thresholds:

    y_pred_threshold = (
        y_prob >= threshold
    ).astype(int)


    precision = precision_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )


    recall = recall_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )


    print(
        f"{threshold:9.2f} | "
        f"{precision:9.3f} | "
        f"{recall:6.3f} | "
        f"{f1:8.3f}"
    )


# ============================================================
# 25. INITIAL RANDOM FOREST
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)


# Random Forest does not require scaled data

rf_model.fit(
    X_train,
    y_train
)

print(
    "\nRandom Forest trained successfully."
)


# Predictions

y_pred_rf = rf_model.predict(
    X_test
)


print("\nFirst 20 Random Forest predictions:")
print(y_pred_rf[:20])


print("\nFirst 20 actual values:")
print(y_test[:20].values)


# ============================================================
# 26. INITIAL RANDOM FOREST EVALUATION
# ============================================================

cm_rf = confusion_matrix(
    y_test,
    y_pred_rf
)

print(
    "\nInitial Random Forest Confusion Matrix:"
)

print(cm_rf)


accuracy_rf = accuracy_score(
    y_test,
    y_pred_rf
)

precision_rf = precision_score(
    y_test,
    y_pred_rf,
    zero_division=0
)

recall_rf = recall_score(
    y_test,
    y_pred_rf,
    zero_division=0
)

f1_rf = f1_score(
    y_test,
    y_pred_rf,
    zero_division=0
)


print(
    "\nAccuracy:",
    accuracy_rf
)

print(
    "Precision:",
    precision_rf
)

print(
    "Recall:",
    recall_rf
)

print(
    "F1-score:",
    f1_rf
)


print(
    "\nInitial Random Forest Classification Report:"
)

print(
    classification_report(
        y_test,
        y_pred_rf,
        zero_division=0
    )
)


# Initial Random Forest ROC-AUC

y_prob_rf = (
    rf_model
    .predict_proba(X_test)[:, 1]
)

roc_auc_rf = roc_auc_score(
    y_test,
    y_prob_rf
)

print(
    "Initial Random Forest ROC-AUC:",
    roc_auc_rf
)


# ============================================================
# 27. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({

    "Feature": X_train.columns,

    "Importance":
        rf_model.feature_importances_

})


feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print(
    "\nRandom Forest Feature Importance:"
)

print(feature_importance)


# ============================================================
# 28. CORRELATION ANALYSIS
# ============================================================

correlation_matrix = model_df[
    [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure"
    ]
].corr()


print("\nCorrelation Matrix:")

print(
    correlation_matrix.round(3)
)


# ============================================================
# 29. 5-FOLD STRATIFIED CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


cv_scores = cross_val_score(
    rf_model,
    X_train,
    y_train,
    cv=cv,
    scoring="roc_auc"
)


print(
    "\nRandom Forest ROC-AUC scores for each fold:"
)

print(cv_scores)


print(
    "\nMean ROC-AUC:",
    cv_scores.mean()
)

print(
    "Standard deviation:",
    cv_scores.std()
)


# ============================================================
# 30. RANDOM FOREST HYPERPARAMETER TUNING
# ============================================================

param_grid = {

    "n_estimators": [
        200,
        300
    ],

    "max_depth": [
        None,
        10,
        20
    ],

    "min_samples_leaf": [
        1,
        2
    ],

    "class_weight": [
        "balanced"
    ]

}


grid_search = GridSearchCV(

    estimator=RandomForestClassifier(
        random_state=42
    ),

    param_grid=param_grid,

    cv=5,

    scoring="roc_auc",

    n_jobs=-1

)


grid_search.fit(
    X_train,
    y_train
)


print(
    "\nBest Random Forest parameters:"
)

print(
    grid_search.best_params_
)


print(
    "\nBest cross-validation ROC-AUC:"
)

print(
    grid_search.best_score_
)


# ============================================================
# 31. TRAIN FINAL TUNED RANDOM FOREST
# ============================================================

best_rf_model = (
    grid_search.best_estimator_
)


best_rf_model.fit(
    X_train,
    y_train
)


print(
    "\nFinal Random Forest trained successfully."
)


# ============================================================
# 32. FINAL RANDOM FOREST EVALUATION
# ============================================================

y_pred_final = (
    best_rf_model.predict(X_test)
)


cm_final = confusion_matrix(
    y_test,
    y_pred_final
)


print(
    "\nFinal Random Forest Confusion Matrix:"
)

print(cm_final)


accuracy_final = accuracy_score(
    y_test,
    y_pred_final
)


precision_final = precision_score(
    y_test,
    y_pred_final,
    zero_division=0
)


recall_final = recall_score(
    y_test,
    y_pred_final,
    zero_division=0
)


f1_final = f1_score(
    y_test,
    y_pred_final,
    zero_division=0
)


print(
    "\nAccuracy:",
    accuracy_final
)

print(
    "Precision:",
    precision_final
)

print(
    "Recall:",
    recall_final
)

print(
    "F1-score:",
    f1_final
)


print(
    "\nFinal Random Forest Classification Report:"
)

print(
    classification_report(
        y_test,
        y_pred_final,
        zero_division=0
    )
)


# ============================================================
# 33. FINAL RANDOM FOREST ROC-AUC
# ============================================================

y_prob_final = (
    best_rf_model
    .predict_proba(X_test)[:, 1]
)


roc_auc_final = roc_auc_score(
    y_test,
    y_prob_final
)


print(
    "\nFinal Random Forest ROC-AUC:",
    roc_auc_final
)


# ============================================================
# 34. FINAL RANDOM FOREST THRESHOLD ANALYSIS
# ============================================================

thresholds_final = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]


print(
    "\nFinal Random Forest threshold analysis:"
)

print(
    "Threshold | Precision | Recall | F1-score"
)

print("-" * 45)


for threshold in thresholds_final:

    y_pred_threshold = (
        y_prob_final >= threshold
    ).astype(int)


    precision = precision_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )


    recall = recall_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )


    print(
        f"{threshold:9.2f} | "
        f"{precision:9.3f} | "
        f"{recall:6.3f} | "
        f"{f1:8.3f}"
    )


# ============================================================
# 35. MODEL ANALYSIS BY TORQUE
# ============================================================

analysis_df = X_test.copy()


analysis_df["Actual failure"] = y_test

analysis_df["Predicted probability"] = (
    y_prob_final
)


analysis_df["Torque group"] = pd.cut(

    analysis_df["Torque [Nm]"],

    bins=[
        0,
        20,
        30,
        40,
        50,
        60,
        70,
        80
    ],

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


torque_model_analysis = (

    analysis_df

    .groupby(
        "Torque group",
        observed=True
    )

    .agg(

        observations=(
            "Actual failure",
            "count"
        ),

        actual_failures=(
            "Actual failure",
            "sum"
        ),

        actual_failure_rate=(
            "Actual failure",
            "mean"
        ),

        mean_predicted_probability=(
            "Predicted probability",
            "mean"
        )

    )
)


torque_model_analysis[
    "actual_failure_rate"
] *= 100


print(
    "\nFinal model analysis by torque:"
)

print(
    torque_model_analysis.round(3)
)


# ============================================================
# 36. MODEL ANALYSIS BY TOOL WEAR
# ============================================================

analysis_df["Tool wear group"] = pd.cut(

    analysis_df["Tool wear [min]"],

    bins=[
        0,
        50,
        100,
        150,
        200,
        250,
        300
    ],

    labels=[
        "0-50",
        "51-100",
        "101-150",
        "151-200",
        "201-250",
        "251-300"
    ]

)


toolwear_model_analysis = (

    analysis_df

    .groupby(
        "Tool wear group",
        observed=True
    )

    .agg(

        observations=(
            "Actual failure",
            "count"
        ),

        actual_failures=(
            "Actual failure",
            "sum"
        ),

        actual_failure_rate=(
            "Actual failure",
            "mean"
        ),

        mean_predicted_probability=(
            "Predicted probability",
            "mean"
        )

    )
)


toolwear_model_analysis[
    "actual_failure_rate"
] *= 100


print(
    "\nFinal model analysis by tool wear:"
)

print(
    toolwear_model_analysis.round(3)
)


# ============================================================
# 37. MODEL COMPARISON
# ============================================================

model_comparison = pd.DataFrame({

    "Model": [

        "Logistic Regression",

        "Random Forest (Initial)",

        "Random Forest (Tuned)"

    ],


    "Accuracy": [

        accuracy_score(
            y_test,
            y_pred
        ),

        accuracy_score(
            y_test,
            y_pred_rf
        ),

        accuracy_score(
            y_test,
            y_pred_final
        )

    ],


    "Precision": [

        precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        precision_score(
            y_test,
            y_pred_rf,
            zero_division=0
        ),

        precision_score(
            y_test,
            y_pred_final,
            zero_division=0
        )

    ],


    "Recall": [

        recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        recall_score(
            y_test,
            y_pred_rf,
            zero_division=0
        ),

        recall_score(
            y_test,
            y_pred_final,
            zero_division=0
        )

    ],


    "F1": [

        f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        f1_score(
            y_test,
            y_pred_rf,
            zero_division=0
        ),

        f1_score(
            y_test,
            y_pred_final,
            zero_division=0
        )

    ],


    "ROC-AUC": [

        roc_auc_score(
            y_test,
            y_prob
        ),

        roc_auc_score(
            y_test,
            y_prob_rf
        ),

        roc_auc_score(
            y_test,
            y_prob_final
        )

    ]

})


print("\nModel comparison:")

print(
    model_comparison.round(4)
)


# ============================================================
# 38. SAVE FINAL MODEL
# ============================================================

model_path = (
    r"C:\Users\EDSON\Documents"
    r"\AI-Apparatus-preservation"
    r"\machine_failure_model.pkl"
)


joblib.dump(
    best_rf_model,
    model_path
)


print(
    "\nModel saved successfully:"
)

print(model_path)


# ============================================================
# 39. LOAD SAVED MODEL
# ============================================================

loaded_model = joblib.load(
    model_path
)


print(
    "\nSaved model loaded successfully."
)


# ============================================================
# 40. TEST SAVED MODEL WITH A NEW MACHINE
# ============================================================

test_machine = pd.DataFrame({

    "Air temperature [K]": [
        300
    ],

    "Process temperature [K]": [
        310
    ],

    "Rotational speed [rpm]": [
        1500
    ],

    "Torque [Nm]": [
        50
    ],

    "Tool wear [min]": [
        200
    ],

    "Type_H": [
        0
    ],

    "Type_L": [
        1
    ],

    "Type_M": [
        0
    ]

})


# Make sure the feature order matches
# the order used during model training.

test_machine = test_machine[
    loaded_model.feature_names_in_
]


# ============================================================
# 41. MAKE PREDICTION
# ============================================================

prediction = loaded_model.predict(
    test_machine
)


print(
    "\nTest machine prediction:",
    prediction
)


# ============================================================
# 42. FAILURE PROBABILITY
# ============================================================

probability = (
    loaded_model
    .predict_proba(test_machine)[0][1]
)


print(
    "Failure probability:",
    probability
)


print(
    "Failure probability (%):",
    probability * 100
)


# ============================================================
# 43. DISPLAY FINAL RESULT
# ============================================================

if prediction[0] == 1:

    print(
        "Result: Machine failure predicted."
    )

else:

    print(
        "Result: No machine failure predicted."
    )

# ============================================================