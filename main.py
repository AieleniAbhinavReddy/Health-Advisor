import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
from typing import List

# Define Pydantic models for request and response
class HealthAdvisorInput(BaseModel):
    age: int
    gender: str
    lifestyle: str
    family_history: str

class HealthAdvisorOutput(BaseModel):
    vaccines: List[str]
    screenings: List[str]
    recommendations: List[str]

# Initialize the FastAPI app
app = FastAPI(
    title="Health Advisor API",
    description="A FastAPI for predicting health recommendations based on personal data."
)

# Load the models and label encoders once when the application starts
models = {}
mlbs = {}

def load_models():
    """Loads all trained models and MultiLabelBinarizer objects from pickle files."""
    try:
        models['vaccines'] = pickle.load(open("./pickles/vaccines_model.pkl", "rb"))
        mlbs['vaccines'] = pickle.load(open("./pickles/vaccines_mlb.pkl", "rb"))
        
        models['screenings'] = pickle.load(open("./pickles/screenings_model.pkl", "rb"))
        mlbs['screenings'] = pickle.load(open("./pickles/screenings_mlb.pkl", "rb"))
        
        models['recommendations'] = pickle.load(open("./pickles/recommendations_model.pkl", "rb"))
        mlbs['recommendations'] = pickle.load(open("./pickles/recommendations_mlb.pkl", "rb"))
        
        # Load the original data to get the correct column order for one-hot encoding
        df_train = pd.read_csv("./datasets/pcd.csv")
        df_train = df_train.drop('input_text', axis=1)
        global trained_columns
        trained_columns = pd.get_dummies(df_train[['age', 'gender', 'lifestyle', 'family_history']],
                                         columns=['gender', 'lifestyle', 'family_history'], drop_first=True).columns.tolist()

    except FileNotFoundError as e:
        raise RuntimeError(f"Error loading model files: {e}. Ensure the pickles and pcd.csv are in the correct directory.")

# Run the model loading function on startup
load_models()

def preprocess_input(input_data: HealthAdvisorInput):
    """Preprocesses a single input request into a format for the models."""
    input_df = pd.DataFrame([input_data.dict()])
    input_dummies = pd.get_dummies(input_df, columns=['gender', 'lifestyle', 'family_history'], drop_first=True)
    
    # Reindex the input DataFrame to match the training data columns, filling missing ones with 0
    aligned_input = input_dummies.reindex(columns=trained_columns, fill_value=0)
    
    return aligned_input

@app.get("/")
def read_root():
    """Home endpoint to check if the API is running."""
    return {"message": "Health Advisor API is running successfully!"}

@app.post("/predict", response_model=HealthAdvisorOutput)
async def predict(input_data: HealthAdvisorInput):
    """
    Accepts four input parameters and returns predictions from the three models.
    """
    processed_input = preprocess_input(input_data)
    
    # Make predictions and inverse transform the results
    vaccines_pred_encoded = models['vaccines'].predict(processed_input)
    vaccines_labels = mlbs['vaccines'].inverse_transform(vaccines_pred_encoded)
    
    screenings_pred_encoded = models['screenings'].predict(processed_input)
    screenings_labels = mlbs['screenings'].inverse_transform(screenings_pred_encoded)
    
    recommendations_pred_encoded = models['recommendations'].predict(processed_input)
    recommendations_labels = mlbs['recommendations'].inverse_transform(recommendations_pred_encoded)
    
    # Construct the final response
    response_data = {
        "vaccines": list(vaccines_labels[0]),
        "screenings": list(screenings_labels[0]),
        "recommendations": list(recommendations_labels[0])
    }
    
    return HealthAdvisorOutput(**response_data)

if __name__ == "__main__":
    # Use the --host 0.0.0.0 for a Docker environment
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))