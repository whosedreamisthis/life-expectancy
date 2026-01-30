import numpy as np
import pandas as pd

from le_package.features import Mapper, TemporalVariableTransformer


def test_temporal_variable_transformer():
    # 1. Arrange
    transformer = TemporalVariableTransformer(
        variables=["YearRemodAdd"], reference_variable="YrSold"
    )
    df = pd.DataFrame({"YearRemodAdd": [2010, 2000], "YrSold": [2015, 2015]})

    # 2. Act
    X = transformer.transform(df)

    # 3. Assert
    assert X["YearRemodAdd"].iat[0] == 5
    assert X["YearRemodAdd"].iat[1] == 15
    assert X.shape == df.shape


def test_mapper_transformer():
    # 1. Arrange
    mappings = {"Gd": 4, "TA": 3, "Fa": 2}
    transformer = Mapper(variables=["ExterQual"], mappings=mappings)
    df = pd.DataFrame({"ExterQual": ["Gd", "TA", "Fa", "Gd"]})

    transformer.fit(df)
    # 2. Act
    X = transformer.transform(df)

    # 3. Assert
    assert X["ExterQual"].iat[0] == 4
    assert X["ExterQual"].iat[2] == 2
    assert not X["ExterQual"].isnull().any()


def test_mapper_transformer_with_missing_keys():
    # Testing how your code handles values not in the mapping
    mappings = {"Gd": 4}
    transformer = Mapper(variables=["ExterQual"], mappings=mappings)
    df = pd.DataFrame({"ExterQual": ["Gd", "MissingValue"]})
    transformer.fit(df)
    X = transformer.transform(df)

    # Based on .map() behavior, unknown values become NaN
    assert np.isnan(X["ExterQual"].iat[1])
