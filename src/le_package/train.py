import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import MeanMedianImputer
from feature_engine.selection import DropFeatures
from feature_engine.transformation import LogCpTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from le_package import config as cfg
from le_package import features as pp

# Force Scikit-Learn to return Pandas DataFrames
sklearn.set_config(transform_output="pandas")


def run_training():
    """Train the model pipeline."""

    # 1. Load Data and pretrain engineering
    data = pd.read_csv(cfg.DATA_DIR / cfg.TRAINING_DATA_FILE)  #
    data.columns = data.columns.str.strip()
    data.dropna(subset=[cfg.TARGET], inplace=True)
    data = data.sort_values(["Country", "Year"])

    X_train = data.drop(cfg.TARGET, axis=1)  #
    y_train = data[cfg.TARGET]  #

    # 2. Setup the Pipeline
    pipe = Pipeline(
        [
            ("continent_gen", pp.ContinentConverter(country_col="Country")),
            (
                "by_country_imputer",
                pp.CountryInterpolator(variables=cfg.NUMERICAL_VARS_WITH_NA),
            ),
            (
                "group_imputer",
                pp.GroupedMedianImputer(variables=cfg.NUMERICAL_VARS_WITH_NA),
            ),
            (
                "final_imputer",
                MeanMedianImputer(
                    imputation_method="median", variables=cfg.NUMERICAL_VARS_WITH_NA
                ),
            ),
            ("imm_score", pp.ImmunizationFeatureCreator(variables=cfg.IMM_VARS)),
            ("binarizer", pp.Binarizer(variables=cfg.BINARIZE_VARS)),
            ("log", LogCpTransformer(variables=cfg.NUMERICAL_LOG_VARS, C=1)),
            ("drop", DropFeatures(features_to_drop=cfg.DROP_FEATURES)),
            ("one_hot", OneHotEncoder(variables=cfg.OHE_VARS, drop_last=True)),
            ("robust_scaler", RobustScaler()),
            (
                "reg",
                RandomForestRegressor(
                    n_estimators=cfg.N_ESTIMATORS, random_state=cfg.SEED
                ),
            ),
        ]
    )

    # 3. Fit the pipeline
    # We call .fit() separately to ensure all custom 'fitted_' flags are set correctly
    pipe.fit(X_train, y_train)  #

    # 4. Save the trained pipeline
    import joblib

    # Using the path structure defined in your config and predict scripts
    filename = f"{cfg.PIPELINE_BASE_FILENAME}_v{cfg.VERSION}.pkl"
    PIPELINE_PATH = cfg.MODEL_DIR / filename
    joblib.dump(pipe, PIPELINE_PATH)
    print(f"Pipeline saved at: {PIPELINE_PATH}")


if __name__ == "__main__":
    run_training()
