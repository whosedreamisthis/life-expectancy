import joblib
import pandas as pd
from le_package import config

def make_prediction():
    # 1. Load the model using the VERSION-based filename
    model_path = config.MODEL_DIR / f"le_model_v{config.VERSION}.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Did you run training first?")
    
    pipe = joblib.load(model_path)

    # 2. Load the data using the path logic from config.py
    data_path = config.DATA_DIR / config.TRAINING_DATA_FILE
    data = pd.read_csv(data_path)

    # 3. Clean columns (Essential for the WHO dataset)
    data.columns = data.columns.str.strip()
    
    # 4. Prepare features (Drop the target if it exists)
    X = data.drop(config.TARGET, axis=1) if config.TARGET in data.columns else data

    # 5. Predict
    results = pipe.predict(X)
    
    print(f"--- Prediction Results (v{config.VERSION}) ---")
    print(f"Predictions for first 5 rows: {results[:20]}")
    return results

if __name__ == "__main__":
    make_prediction()