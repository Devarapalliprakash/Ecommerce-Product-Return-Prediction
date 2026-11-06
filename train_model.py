
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# STEP 1: Load dataset
df = pd.read_csv(
    "dataset/Section7_Q4_Ecommerce_Product_Return_CICD.csv"
)

# STEP 2: Remove duplicate records
df = df.drop_duplicates()

# STEP 3: Handle invalid values
df.loc[df["Product_Price_INR"] <= 0, "Product_Price_INR"] = np.nan

df.loc[
    ~df["Discount_Percent"].between(0, 100),
    "Discount_Percent"
] = np.nan

df.loc[
    ~df["Customer_Rating_PreOrder"].between(1, 5),
    "Customer_Rating_PreOrder"
] = np.nan

df.loc[df["Actual_Delivery_Days"] < 0, "Actual_Delivery_Days"] = np.nan

df.loc[df["Previous_Returns"] < 0, "Previous_Returns"] = np.nan

# STEP 4: Feature Engineering
df["Discount_Amount_INR"] = (
    df["Product_Price_INR"] *
    df["Discount_Percent"] / 100
)

df["Delivery_Delay_Days"] = (
    df["Actual_Delivery_Days"] -
    df["Expected_Delivery_Days"]
)

df["Customer_Return_Frequency"] = np.where(
    df["Previous_Orders"] > 0,
    df["Previous_Returns"] / df["Previous_Orders"],
    0
)

# STEP 5: Prepare input and output
X = df.drop(columns=["Return_Status", "Order_ID"])
y = df["Return_Status"].map({
    "Not Returned": 0,
    "Returned": 1
})

# Remove records with missing target
valid_rows = y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows]

# STEP 6: Identify column types
numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

# STEP 7: Preprocessing
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])

# STEP 8: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("TRAINING RECORDS:", len(X_train))
print("TESTING RECORDS:", len(X_test))

# STEP 9: Create two models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
}

results = {}
trained_models = {}

# STEP 10: Train and evaluate models
for name, model in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    print("\n==============================")
    print("MODEL:", name)
    print("==============================")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy
    trained_models[name] = pipeline

    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Not Returned", "Returned"],
        zero_division=0
    ))

# STEP 11: Select best model
best_model_name = max(results, key=results.get)

best_model = trained_models[best_model_name]

# STEP 12: Save best model
joblib.dump(
    best_model,
    "best_return_model.pkl"
)

print("\n========== MODEL COMPARISON ==========")

for name, accuracy in results.items():
    print(name, ":", round(accuracy * 100, 2), "%")

print("\nBEST MODEL:", best_model_name)
print("BEST ACCURACY:", round(results[best_model_name] * 100, 2), "%")

print("\nBEST MODEL SAVED SUCCESSFULLY!")
