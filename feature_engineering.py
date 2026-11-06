
import pandas as pd
import numpy as np

# STEP 1: Load original dataset
df = pd.read_csv(
    "dataset/Section7_Q4_Ecommerce_Product_Return_CICD.csv"
)

print("ORIGINAL DATASET")
print(df.head())

# STEP 2: Handle missing numerical values
numerical_columns = [
    "Product_Price_INR",
    "Discount_Percent",
    "Actual_Delivery_Days",
    "Customer_Rating_PreOrder"
]

for col in numerical_columns:
    df[col] = df[col].fillna(df[col].median())

# STEP 3: Handle missing categorical values
df["Payment_Method"] = df["Payment_Method"].fillna(
    df["Payment_Method"].mode()[0]
)

# STEP 4: Correct invalid product prices
valid_price = df.loc[
    df["Product_Price_INR"] > 0,
    "Product_Price_INR"
].median()

df.loc[
    df["Product_Price_INR"] <= 0,
    "Product_Price_INR"
] = valid_price

# STEP 5: Correct invalid discounts
valid_discount = df.loc[
    df["Discount_Percent"].between(0, 100),
    "Discount_Percent"
].median()

df.loc[
    ~df["Discount_Percent"].between(0, 100),
    "Discount_Percent"
] = valid_discount

# STEP 6: FEATURE ENGINEERING

# Feature 1: Discount Amount
df["Discount_Amount_INR"] = (
    df["Product_Price_INR"] *
    df["Discount_Percent"] / 100
)

# Feature 2: Delivery Delay
df["Delivery_Delay_Days"] = (
    df["Actual_Delivery_Days"] -
    df["Expected_Delivery_Days"]
)

# Feature 3: Customer Return Frequency
df["Customer_Return_Frequency"] = np.where(
    df["Previous_Orders"] > 0,
    df["Previous_Returns"] / df["Previous_Orders"],
    0
)

# STEP 7: Display newly created features
print("\nNEW FEATURES CREATED")

print(df[[
    "Discount_Amount_INR",
    "Delivery_Delay_Days",
    "Customer_Return_Frequency"
]].head(10))

# STEP 8: Display final dataset shape
print("\nFINAL DATASET SHAPE")
print(df.shape)

# STEP 9: Check missing values
print("\nTOTAL MISSING VALUES")
print(df.isnull().sum().sum())

# STEP 10: Save engineered dataset
df.to_csv(
    "dataset/engineered_ecommerce_data.csv",
    index=False
)

print("\nFEATURE ENGINEERING COMPLETED!")
print("Engineered dataset saved successfully!")
