import numpy as np
import pandas as pd
import pytest

from le_package.features import (
    Binarizer,
    ContinentConverter,
    CountryInterpolator,
    GroupedMedianImputer,
    ImmunizationFeatureCreator,
)


def test_country_interpolator():
    # 1. Arrange: Create data with a gap for a specific country
    df = pd.DataFrame(
        {
            "Country": ["Afghanistan", "Afghanistan", "Afghanistan"],
            "BMI": [20.0, np.nan, 22.0],
        }
    )
    transformer = CountryInterpolator(variables=["BMI"])

    # 2. Act
    X = transformer.transform(df)

    # 3. Assert: Check if the middle value was interpolated to 21.0
    assert X["BMI"].iat[1] == 21.0


def test_grouped_median_imputer():
    # 1. Arrange: Status-based medians are 50 for Developing, 80 for Developed
    df = pd.DataFrame(
        {
            "Status": ["Developing", "Developing", "Developed", "Developed"],
            "Schooling": [50, np.nan, 80, np.nan],
        }
    )
    transformer = GroupedMedianImputer(variables=["Schooling"])

    # 2. Act
    transformer.fit(df)
    X = transformer.transform(df)

    # 3. Assert
    assert X["Schooling"].iat[1] == 50.0
    assert X["Schooling"].iat[3] == 80.0


def test_binarizer():
    # 1. Arrange
    df = pd.DataFrame({"HIV/AIDS": [0.05, 0.5, 0.0, 1.2]})
    transformer = Binarizer(variables=["HIV/AIDS"], threshold=0.1)

    # 2. Act
    X = transformer.transform(df)

    # 3. Assert: 0.05 is below 0.1 (-> 0), others are above (-> 1)
    assert X["HIV/AIDS"].iat[0] == 0
    assert X["HIV/AIDS"].iat[1] == 1


def test_continent_converter():
    # 1. Arrange
    df = pd.DataFrame({"Country": ["Canada", "Germany", "Japan"]})
    transformer = ContinentConverter(country_col="Country")

    # 2. Act
    X = transformer.transform(df)

    # 3. Assert: Check if continent mapping via country_converter works
    assert X["Continent"].iat[0] == "America"
    assert X["Continent"].iat[1] == "Europe"
    assert X["Continent"].iat[2] == "Asia"


def test_immunization_feature_creator():
    # 1. Arrange
    cols = ["Polio", "Diphtheria", "Hepatitis B"]
    df = pd.DataFrame(
        {"Polio": [100, 80], "Diphtheria": [90, 70], "Hepatitis B": [80, 60]}
    )
    transformer = ImmunizationFeatureCreator(variables=cols)

    # 2. Act
    X = transformer.transform(df)

    # 3. Assert: Mean of (100, 90, 80) is 90
    assert X["Immunization_Score"].iat[0] == 90.0
    assert "Immunization_Score" in X.columns
