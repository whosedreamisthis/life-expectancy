# Life Expectancy Prediction Pipeline

This project implements a machine learning pipeline to predict Life Expectancy based on WHO datasets. It utilizes a modular package structure, custom Scikit-Learn transformers, and automated testing via Tox.

## 📂 Project Structure

- **src/le_package/**: Core package containing the ML logic.
  - `config.py`: Centralized configuration and path management.
  - `features.py`: Custom Scikit-Learn transformers (e.g., ContinentConverter, CountryInterpolator).
  - `train.py`: Script to build, fit, and save the pipeline.
  - `predict.py`: Script to load the trained model and run inference.
  - `VERSION`: File containing the current package version.
  - `configs/params.yaml`: YAML file for model hyperparameters and feature lists.
- **data/**: Directory containing the `Life Expectancy Data.csv`.
- **models/**: Directory where trained `.pkl` model artifacts are stored.
- **tests/**: Test suite for verifying transformers and training logic.
- **tox.ini**: Configuration for automated environment setup and testing.

## 🚀 Features

- **Custom Transformers**: Includes `ContinentConverter` for geographic feature engineering and `CountryInterpolator` for handling missing time-series data.
- **Automated Pipeline**: A full Scikit-Learn `Pipeline` that handles imputation, binarization, log transformation, and scaling before reaching the `RandomForestRegressor`.
- **Versioned Artifacts**: Models are saved with version suffixes (e.g., `le_model_v0.1.0.pkl`) to match the project version.
- **Path Robustness**: The configuration logic automatically detects if the code is running in a local dev environment or an installed `tox` site-packages environment.

## 🛠️ Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt