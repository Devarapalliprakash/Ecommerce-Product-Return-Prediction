
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib

# Create FastAPI application
app = FastAPI(
    title="E-Commerce Product Return Prediction",
    description="Machine Learning API for product return prediction",
    version="1.0"
)

# Load trained ML model
model = joblib.load("best_return_model.pkl")


# Input data structure
class ProductData(BaseModel):

    Product_Category: str
    Product_Price_INR: float = Field(gt=0)
    Discount_Percent: float = Field(ge=0, le=100)

    Expected_Delivery_Days: int = Field(ge=0)
    Actual_Delivery_Days: float = Field(ge=0)

    Customer_Rating_PreOrder: float = Field(ge=1, le=5)

    Previous_Returns: int = Field(ge=0)
    Previous_Orders: int = Field(ge=0)

    Payment_Method: str
    Damaged_Package_Flag: str
    Package_Size: str
    Promotion_Active: str


# Health endpoint
@app.get("/health")
def health():

    return {
        "status": "Application is running",
        "model_loaded": True
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: ProductData):

    # Convert input into dictionary
    input_data = data.model_dump()

    # Feature engineering
    input_data["Discount_Amount_INR"] = (
        input_data["Product_Price_INR"] *
        input_data["Discount_Percent"] / 100
    )

    input_data["Delivery_Delay_Days"] = (
        input_data["Actual_Delivery_Days"] -
        input_data["Expected_Delivery_Days"]
    )

    if input_data["Previous_Orders"] > 0:

        input_data["Customer_Return_Frequency"] = (
            input_data["Previous_Returns"] /
            input_data["Previous_Orders"]
        )

    else:
        input_data["Customer_Return_Frequency"] = 0

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Predict using trained model
    prediction = int(model.predict(input_df)[0])

    # Convert prediction into text
    result = (
        "Returned" if prediction == 1
        else "Not Returned"
    )

    return {
        "prediction": result,
        "prediction_code": prediction,
        "message": "Product return prediction completed successfully"
    }
