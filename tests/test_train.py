import os
from pathlib import Path
import joblib
import pandas as pd
from le_package import config
from le_package.train import run_training

def test_run_training_saves_model():
    """
    Integration test to ensure the training pipeline runs
    and saves a model file to the disk.
    """
    # 1. Arrange
    # Define the exact filename that train.py is expected to produce
    save_file_name = config.MODEL_DIR / f"{config.PIPELINE_BASE_FILENAME}_v{config.VERSION}.pkl"

    # Remove old model if it exists to ensure a clean test
    if save_file_name.exists():
        save_file_name.unlink()

    # 2. Act
    run_training()

    # 3. Assert
    assert save_file_name.exists()
    assert os.path.getsize(save_file_name) > 0

def test_saved_model_is_loadable():
    """
    Ensures the saved pipeline can be loaded and perform a prediction.
    """
    # 1. Arrange
    save_file_name = config.MODEL_DIR / f"{config.PIPELINE_BASE_FILENAME}_v{config.VERSION}.pkl"
    
    # 2. Act
    if not save_file_name.exists():
        run_training()
        
    loaded_pipe = joblib.load(save_file_name)
    
    # Create a tiny dummy sample matching the expected input structure
    sample_data = pd.DataFrame({
        'Country': ['Afghanistan'],
        'Year': [2015],
        'Status': ['Developing'],
        'Adult Mortality': [263.0],
        'infant deaths': [62],
        'Alcohol': [0.01],
        'percentage expenditure': [71.27],
        'Hepatitis B': [65.0],
        'Measles': [1154],
        'BMI': [19.1],
        'under-five deaths': [83],
        'Polio': [6.0],
        'Total expenditure': [8.16],
        'Diphtheria': [65.0],
        'HIV/AIDS': [0.1],
        'GDP': [584.25],
        'Population': [33736494.0],
        'thinness  1-19 years': [17.2],
        'thinness 5-9 years': [17.3],
        'Income composition of resources': [0.479],
        'Schooling': [10.1]
    })

    # 3. Assert
    # Test that the pipeline successfully processes data and returns a float
    prediction = loaded_pipe.predict(sample_data)
    assert isinstance(prediction[0], float)