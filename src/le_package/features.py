import country_converter as coco
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CountryInterpolator(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        # Nothing to learn here, just return self
        return self

    def transform(self, X):
        X_copy = X.copy()

        X_copy[self.variables] = X_copy.groupby("Country")[self.variables].transform(
            lambda x: x.interpolate(limit_direction="both")
        )
        return X_copy


class GroupedMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
        self.group_values_ = {}

    def fit(self, X, y=None):
        # Calculate and store the median for each Status group
        for var in self.variables:
            self.group_values_[var] = X.groupby("Status")[var].median().to_dict()
        return self

    def transform(self, X):
        X_copy = X.copy()
        for var in self.variables:
            # Map the calculated medians back to the rows based on Status
            fill_values = X_copy["Status"].map(self.group_values_[var])
            X_copy[var] = X_copy[var].fillna(fill_values)
        return X_copy


class Binarizer(BaseEstimator, TransformerMixin):
    def __init__(self, variables, threshold=0.1):
        self.variables = variables
        self.threshold = threshold

    def fit(self, X, y=None):
        # Nothing to learn from the data
        return self

    def transform(self, X):
        X_copy = X.copy()
        for var in self.variables:
            X_copy[var] = (X_copy[var] > self.threshold).astype(int)
        return X_copy


def get_continent(df, column):
    # This keeps the coco logic inside a standard function call
    return coco.convert(names=df[column], to="continent")


class ContinentConverter(BaseEstimator, TransformerMixin):
    def __init__(self, country_col):
        self.country_col = country_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy["Continent"] = get_continent(X_copy, self.country_col)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return np.append(input_features, "Continent")


class ImmunizationFeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Calculate the mean across the specified immunization columns
        X_copy["Immunization_Score"] = X_copy[self.variables].mean(axis=1)
        return X_copy
