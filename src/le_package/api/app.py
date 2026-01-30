from fastapi import FastAPI, HTTPException
from le_package.api.schemas import PredictionInput
from le_package.predict import make_prediction
import pandas as pd

import logging
from le_package import config

# Standard logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE),  # This writes to PROJECT_ROOT/logs/api.log
        logging.StreamHandler()                # This keeps showing logs in your terminal
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Life Expectancy API", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    logger.info(f"API called with: {input_data}")
    # Convert Pydantic model to DataFrame for the pipeline
    data_df = pd.DataFrame([input_data.model_dump(by_alias=True)])
    
    try:
        results = make_prediction(input_data=data_df)
        return {"prediction": results[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))