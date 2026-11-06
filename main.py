
import pandas as pd

# STEP 1: Load dataset
df = pd.read_csv(
    "dataset/Section7_Q4_Ecommerce_Product_Return_CICD.csv"
)

# STEP 2: Handle missing numerical values
numerical_columns = [
    "Product_Price_INR",
    "Discount_Percent",
    "Actual_Delivery_Days",
    "Customer_Rating_PreOrder"
]

for col in numerical_columns:
    df[col] = df[col].fillna(df[col].median())

# Handle missing categorical values
df["Payment_Method"] = df["Payment_Method"].fillna(
    df["Payment_Method"].mode()[0]
)

# STEP 3: Handle invalid product prices
valid_price = df.loc[
    df["Product_Price_INR"] > 0,
    "Product_Price_INR"
].median()

df.loc[
    df["Product_Price_INR"] <= 0,
    "Product_Price_INR"
] = valid_price

# STEP 4: Handle invalid discounts
valid_discount = df.loc[
    df["Discount_Percent"].between(0, 100),
    "Discount_Percent"
].median()

df.loc[
    ~df["Discount_Percent"].between(0, 100),
    "Discount_Percent"
] = valid_discount

# STEP 5: Convert target column into numbers
df["Return_Status"] = df["Return_Status"].map({
    "Not Returned": 0,
    "Returned": 1
})

# STEP 6: Remove Order_ID
df = df.drop(columns=["Order_ID"])

# STEP 7: Apply One-Hot Encoding
categorical_columns = [
    "Product_Category",
    "Payment_Method",
    "Damaged_Package_Flag",
    "Package_Size",
    "Promotion_Active"
]

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

# STEP 8: Display results
print("\nENCODED DATASET")
print(df.head())

print("\nNEW COLUMN NAMES")
print(df.columns.tolist())

print("\nDATASET SHAPE AFTER ENCODING")
print(df.shape)

print("\nMISSING VALUES")
print(df.isnull().sum().sum())

# STEP 9: Save encoded dataset
df.to_csv(
    "dataset/encoded_ecommerce_data.csv",
    index=False
)

print("\nCATEGORICAL ENCODING COMPLETED!")
print("Encoded dataset saved successfully!")

