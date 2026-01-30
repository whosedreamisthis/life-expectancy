import joblib
import pandas as pd
from le_package import config

def make_prediction(input_data=None):
    """
    Makes a prediction using the saved model pipeline.
    Accepts a DataFrame or defaults to the training data.
    """
    # 1. Load the model using the VERSION-based filename
    model_path = config.MODEL_DIR / f"le_model_v{config.VERSION}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Did you run training first?"
        )

    pipe = joblib.load(model_path)

    # 2. Determine which data to use
    if input_data is None:
        # Load the data using the path logic from config.py for testing
        data_path = config.DATA_DIR / config.TRAINING_DATA_FILE
        data = pd.read_csv(data_path)
    else:
        # Use the data provided by the API
        data = input_data

    # 3. Clean columns (Essential for the WHO dataset)
    # This handles extra spaces in CSV headers or JSON keys
    data.columns = data.columns.str.strip()

    # 4. Prepare features (Drop the target if it exists in the data)
    X = data.drop(config.TARGET, axis=1) if config.TARGET in data.columns else data

    # 5. Predict
    results = pipe.predict(X)

    return results

if __name__ == "__main__":
    # When running as a script, it will still output test results
    predictions = make_prediction()
    print(f"--- Prediction Results (v{config.VERSION}) ---")
    print(f"First 5 predictions: {predictions[:5]}")