import toolz as tz
import pandas as pd
import numpy as np
from scipy.stats import zscore
import operator as op
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


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
    not_empty = tz.complement(op.attrgetter("empty"))
    split_data = list(filter(not_empty, np.array_split(data, 10)))

    get_training_and_test_data = tz.partial(extract_training_and_test_data, split_data)
    return list(map(get_training_and_test_data, enumerate(split_data)))


@tz.curry
def average_accuracy_and_f1_score(datasets, num_of_neighbours):
    """
    @param datasets: a collection of datasets, each containing training and validation data
    to use for the model
    @param num_of_neighbours: the number of neighbours to compare against
    @return: the average accuracy and f1 score of a K-nearest neighbours model applied on each
    dataset, where k = num_of_neighbours

    """
    def accuracy_and_f1_score(dataset):
        """
        @param dataset: a dataset containing training and validation data
        @return: the accuracy and f1 score for a KNN model trained on the training data
        and comparing with num_of_neighbours many neighbours
        """
        training_dataset, test_dataset = dataset
        training_data, training_mileage_labels = (training_dataset.drop("has_good_mpg", axis='columns'),
                                                  training_dataset['has_good_mpg'])

        test_data, test_mileage_labels = (test_dataset.drop("has_good_mpg", axis='columns'),
                                          test_dataset['has_good_mpg'])

        model = (KNeighborsClassifier(n_neighbors=num_of_neighbours)
                 .fit(training_data, training_mileage_labels))

        actual_mileage_labels = model.predict(test_data)
        return (model.score(test_data, test_mileage_labels),
                f1_score(test_mileage_labels, actual_mileage_labels))

    return tz.thread_last(datasets,
                          (map, accuracy_and_f1_score),
                          list,
                          tz.partial(np.average, axis=0))


if __name__ == '__main__':

    wanted_features = ["cylinders",
                       "displacement",
                       "horsepower",
                       "weight",
                       "acceleration",
                       "mpg"]

    cross_validation_data = (read_car_data('auto-mpg-regression.tsv')
                             .pipe(filter_features, wanted_features)
                             .pipe(add_mileage_label)
                             .drop('mpg', axis="columns")
                             .pipe(scale_features)
                             .pipe(generate_10_fold_cross_validation_data))

    averages = list(map(average_accuracy_and_f1_score(cross_validation_data), [3, 6, 10, 16, 25]))
    print(averages)


