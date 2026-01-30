from fastapi import FastAPI, HTTPException
from le_package.api.schemas import PredictionInput
from le_package.predict import make_prediction
import pandas as pd

app = FastAPI(title="Life Expectancy API", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    # Convert Pydantic model to DataFrame for the pipeline
    data_df = pd.DataFrame([input_data.model_dump(by_alias=True)])
    
    try:
        results = make_prediction(input_data=data_df)
        return {"prediction": results[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))