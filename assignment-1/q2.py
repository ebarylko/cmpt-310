import toolz as tz
import pandas as pd
import numpy as np
from scipy.stats import zscore


def read_car_data(filename):
    return pd.read_csv(filename, sep='\t')


def filter_features(df: pd.DataFrame, wanted_features):
    """
    @param df: a DataFrame
    @param wanted_features: the feature columns to keep
    @return: a DataFrame only containing the columns in df which are in wanted_features
    """
    return df.filter(items=wanted_features)


def add_mileage_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    @param df: a DataFrame where each row contains information about cars, such as horsepower, mpg,
    acceleration, and more info
    @return: adds a column to df which notes whether a given car has a poor or good mileage.
    """
    mpg_median = df['mpg'].median()
    return df.assign(has_good_mpg=df['mpg'] >= mpg_median)


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    @param df: a DataFrame containing information about cars from the 1970s, such as mpg,
    acceleration, and displacement
    @return: a DataFrame having all the features except the mileage label scaled
    """
    cpy = df.copy()
    features_except_mpg, mpg_column = cpy.drop('has_good_mpg', axis="columns"), cpy['has_good_mpg']

    scaled_features = zscore(features_except_mpg)
    return pd.concat([scaled_features, mpg_column], axis=1)


def extract_training_and_test_data(partitioned_data: list[pd.DataFrame], current_test_set):
    """
    @param partitioned_data: a DataFrame split into ten groups
    @param current_test_set: the current partition of partitioned_data considered as the test data
    and its index within the split
    @return: the training data associated with current_test_set and the current test data
    """
    idx, test_data = current_test_set
    training_data = pd.concat(partitioned_data[:idx] + partitioned_data[idx + 1:])
    return training_data, test_data


def generate_10_fold_cross_validation_data(data: pd.DataFrame):
    """
    @param data: a DataFrame where each row contains the mpg,
    acceleration, and displacement of a car among other information
    @return: splits the data into ten disjoint test dataset and pairs each with
    the corresponding training dataset
    """
    split_data = list(filter(lambda df: not df.empty, np.array_split(data, 10)))
    get_training_and_test_data = tz.partial(extract_training_and_test_data, split_data)
    return list(map(get_training_and_test_data, enumerate(split_data)))

