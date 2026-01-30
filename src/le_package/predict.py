import joblib
import pandas as pd
import numpy as np

from le_package.config import MODEL_DIR

# Define the path to the saved model
with open("VERSION", "r") as f:
    VERSION = f.read().strip()
    
PIPELINE_SAVE_FILE = f"life_expectancy_v{VERSION}.joblib"
PIPELINE_PATH = os.path.join(MODEL_DIR, PIPELINE_SAVE_FILE)


def make_prediction(input_data) -> np.ndarray:
    """
    Make a prediction using the saved model pipeline.

    Args:
        input_data: A pandas DataFrame containing the raw input features.
    Returns:
        A list of predictions (or a dictionary with more metadata).
    """

    # 1. Load the persisted pipeline
    trained_model = joblib.load(PIPELINE_PATH)

    # 2. Make predictions
    # Note: If your notebook used log transformation on the target,
    # you might need to apply np.exp() here to return real prices.
    predictions = trained_model.predict(input_data)

    return predictions


if __name__ == "__main__":
    # Small test for local debugging
    test_data = pd.read_csv("data/Life Expectancy Data.csv")
    results = make_prediction(test_data)
    print(f"Sample Predictions: {results[:5]}")
