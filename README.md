# Life Expectancy Prediction Package

This package provides a comprehensive machine learning pipeline for predicting life expectancy using World Health Organization (WHO) data. It leverages **Scikit-learn** and **Feature-engine** to automate end-to-end data preprocessing, feature engineering, and modeling.

---

## Features

The pipeline includes several custom and automated engineering steps:
* **Continent Mapping**: Automatically derives continents from country names using `country-converter`.
* **Intelligent Imputation**: 
    * **Interpolation**: Fills gaps in time-series data for each country (handling missing years).
    * **Grouped Medians**: Fills remaining missing values based on a country's development status (Developed vs. Developing).
* **Feature Engineering**: 
    * Creates an **Immunization Score** by averaging Hepatitis B, Polio, and Diphtheria rates.
    * Binarizes high-risk variables like HIV/AIDS based on clinical thresholds.
    * Applies log transformations to skewed variables (e.g., GDP, Population, Measles).
* **Scalable Architecture**: Uses a Scikit-learn `Pipeline` to ensure consistency between training and inference.

---

## Project Structure

```text
├── src/le_package/
│   ├── configs/
│   │   └── params.yaml      # Model hyperparameters and feature lists
│   ├── config.py            # Path management and YAML loading logic
│   ├── features.py          # Custom Scikit-learn transformers
│   ├── train.py             # Script to train and save the model
│   ├── predict.py           # Script to load the model and generate predictions
│   └── VERSION              # Current package version string
├── tests/                   # Pytest suite for features and training
├── data/                    # Directory for raw CSV data
├── models/                  # Directory where trained .pkl artifacts are stored
├── pyproject.toml           # Build system and dependency configuration
├── tox.ini                  # Automation for testing and linting
└── requirements.txt         # Core dependencies for the environment

## Installation

### Prerequisites
* Python $\ge$ 3.9

### Setup
1. Clone the repository and navigate to the root directory.
2. Install the package in editable mode with dependencies:
   ```bash
   pip install -e .

## Usage

### Configuration
The model behavior is controlled by `src/le_package/configs/params.yaml`. You can modify this file to change the `random_state`, `n_estimators`, or which features to drop/transform without modifying the source code.

### Training the Model
To train the pipeline and save the model artifact to the `models/` folder:

```bash
python src/le_package/train.py

### Making Predictions
The prediction script automatically loads the model version specified in the `VERSION` file:

```bash
python src/le_package/predict.py

## Development & Quality Assurance

### Testing
This project uses `pytest` for unit and integration testing.

```bash
# Run all tests
pytest tests/

### Automation with Tox
You can run the entire suite (training, testing, linting, and type checking) across isolated environments using tox:

```bash
# Run tests and training in a clean env
tox

# Run linting only (Black, Isort, Flake8)
tox -e lint

# Run type checks
tox -e typechecks

### Versioning
The package version is managed in `src/le_package/VERSION`. This version is used to name model artifacts (e.g., `le_model_v0.1.0.pkl`), ensuring that production models are always traceable to specific code versions.

---

## License
[Insert License Information Here]