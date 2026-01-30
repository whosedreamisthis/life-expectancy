import os
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
# Check if we are in a 'tox' environment
if "site-packages" in str(PACKAGE_ROOT):
    # In Tox, the project root is usually the Current Working Directory
    # because that's where you ran the 'tox' command from.
    PROJECT_ROOT = Path(os.getcwd())
else:
    # In local development, use the grandparent of src/le_package
    PROJECT_ROOT = PACKAGE_ROOT.parent.parent

CONFIG_FILE_PATH = PACKAGE_ROOT / "configs" / "params.yaml"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
VERSION_FILE_PATH = PACKAGE_ROOT / "VERSION"


def fetch_version() -> str:
    print("PACKAGE_ROOT ", PACKAGE_ROOT)
    print("PROJECT_ROOT ", PROJECT_ROOT)
    print("VERSION_FILE_PATH ", VERSION_FILE_PATH)
    """Reads the version string from the VERSION file."""
    with open(VERSION_FILE_PATH, "r") as v_file:
        return v_file.read().strip()


# Expose the version variable
VERSION = fetch_version()


# 2. Function to load YAML
def fetch_config_from_yaml(cfg_path: Path = CONFIG_FILE_PATH) -> dict:
    """Parse YAML containing the package configuration."""
    if cfg_path.exists():
        with open(cfg_path, "r") as conf_file:
            parsed_config = yaml.safe_load(conf_file)
            return parsed_config

    raise OSError(f"Did not find config file at path: {cfg_path.absolute()}")


# 3. Initialize Config
_config = fetch_config_from_yaml()
# 4. Map YAML values to variables
# Reproducibility
SEED = _config["random_state"]

# Target
TARGET = _config["target"]

# Features to drop
DROP_FEATURES = _config["features_to_drop"]

# Imputation
NUMERICAL_VARS_WITH_NA = _config["num_vars_with_na"]

# Variables for Transformation
NUMERICAL_LOG_VARS = _config["numerical_log_vars"]
BINARIZE_VARS = _config["binarize_vars"]


OHE_VARS = _config["ohe_vars"]
IMM_VARS = _config["imm_vars"]


    
# Model Hyperparameters
N_ESTIMATORS = _config["n_estimators"]

# Artifact Names

PIPELINE_BASE_FILENAME = "le_model"
PIPELINE_SAVE_FILE = _config.get("pipeline_save_file", "le_model.pkl")
TRAINING_DATA_FILE = _config.get("training_data_file", "Life Expectancy Data.csv")
