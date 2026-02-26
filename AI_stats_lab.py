"""
Linear & Logistic Regression Lab

Follow the instructions in each function carefully.
DO NOT change function names.
Use random_state=42 everywhere required.
"""

import numpy as np

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# QUESTION 1 – Linear Regression Pipeline (Diabetes)
# =========================================================

def diabetes_linear_pipeline():

    # STEP 1
    data = load_diabetes()
    X = data.data
    y = data.target

    # STEP 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # STEP 4
    model = LinearRegression()
    model.fit(X_train, y_train)

    # STEP 5
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # STEP 6
    coef_abs = np.abs(model.coef_)
    top_3_feature_indices = list(np.argsort(coef_abs)[-3:][::-1])

    return (
        train_mse,
        test_mse,
        train_r2,
        test_r2,
        top_3_feature_indices
    )


# =========================================================
# QUESTION 2 – Cross-Validation (Linear Regression)
# =========================================================

def diabetes_cross_validation():

    # STEP 1
    data = load_diabetes()
    X = data.data
    y = data.target

    # STEP 2 (manual pipeline logic)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # STEP 3
    model = LinearRegression()
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')

    # STEP 4
    mean_r2 = np.mean(scores)
    std_r2 = np.std(scores)

    return mean_r2, std_r2


# =========================================================
# QUESTION 3 – Logistic Regression Pipeline (Cancer)
# =========================================================

def cancer_logistic_pipeline():

    # STEP 1
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # STEP 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # STEP 4
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    # STEP 5
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    # Confusion matrix (computed but not returned)
    cm = confusion_matrix(y_test, y_test_pred)

    # False Negative (medical meaning):
    # A False Negative means the model predicts "benign"
    # but the patient actually has cancer.
    # This is dangerous because the disease goes untreated.

    return (
        train_accuracy,
        test_accuracy,
        precision,
        recall,
        f1
    )


# =========================================================
# QUESTION 4 – Logistic Regularization Path
# =========================================================

def cancer_logistic_regularization():

    # STEP 1
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # STEP 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = {}

    # STEP 4
    for C_value in [0.01, 0.1, 1, 10, 100]:
        model = LogisticRegression(max_iter=5000, C=C_value)
        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        results[C_value] = (train_acc, test_acc)

    # Comments:
    # When C is very small → strong regularization → simpler model → may underfit.
    # When C is very large → weak regularization → complex model → may overfit.
    # Overfitting usually happens when C is very large.

    return results


# =========================================================
# QUESTION 5 – Cross-Validation (Logistic Regression)
# =========================================================

def cancer_cross_validation():

    # STEP 1
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # STEP 2
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # STEP 3
    model = LogisticRegression(C=1, max_iter=5000)
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

    # STEP 4
    mean_accuracy = np.mean(scores)
    std_accuracy = np.std(scores)

    # Cross-validation is especially important in medical diagnosis
    # because it ensures the model generalizes well and does not
    # perform well only on one specific train-test split.
    # In medical problems, wrong predictions can cost lives.

    return mean_accuracy, std_accuracy
