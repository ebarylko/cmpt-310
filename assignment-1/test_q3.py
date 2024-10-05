import q3
import q3 as q
import pandas as pd
import operator as op
import functools as ft
import pandas.testing as pdt

sample_data = pd.DataFrame({"mpg": [1, 2, 3, 4],
                            "displacement": [3, 4, 5, 6]})

feature_to_scaling_function = [("mpg", ft.partial(op.add, 1)),
                               ("displacement", ft.partial(op.mul, 2))]

expected = pd.DataFrame({"mpg": [2, 3, 4, 5],
                         "displacement": [6, 8, 10, 12]})


def test_normalize_and_one_hot_encode_data():
    pdt.assert_frame_equal(q.normalize_and_one_hot_encode_data(sample_data, feature_to_scaling_function), expected)
#
#
# sample = pd.Series([1, 2, 3, 3])
#
# expected_encoding = pd.DataFrame([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], columns=[1, 2, 3])
#
#
# def test_one_hot():
#     pdt.assert_frame_equal(expected_encoding, q.one_hot(sample), check_dtype=False)


sample_car_data = pd.DataFrame({"mpg": [1, 2, 3],
                                "acceleration": [5, 6, 8]})

expected_data = pd.DataFrame({"mpg": [1, 2, 3],
                              "acceleration": [5, 6, 8],
                              "has_good_mpg": [False, True, True]})


def test_has_good_mileage():
    pdt.assert_frame_equal(q.add_mileage_label(sample_car_data), expected_data, check_like=False)


sample = pd.DataFrame({"mpg": [1, 2, 3],
                       "has_good_mpg": [False, True, True]})

expected_scaled_data = pd.DataFrame({"mpg": [-1.22474487, 0, 1.22474487],
                                     "has_good_mpg": [False, True, True]})


def test_scale_features():
    pdt.assert_frame_equal(q3.scale_features(sample), expected_scaled_data)


sample_info = pd.DataFrame({"mpg": [1, 2, 3],
                            "acceleration": [3, 5, 6],
                            "has_good_mpg": [False, True, True]})

expected_first_training_set, expected_first_test_set = (pd.DataFrame({"mpg": [2, 3],
                                                                      "acceleration": [5, 6],
                                                                      "has_good_mpg": [True, True]},
                                                                     index=[1, 2]), pd.DataFrame({"mpg": [1],
                                                                                                  "acceleration": [3],
                                                                                                  "has_good_mpg": [False]}))

expected_second_training_set, expected_second_test_set  = (pd.DataFrame({"mpg": [1, 3],
                                    "acceleration": [3, 6],
                                     "has_good_mpg": [False, True]}), pd.DataFrame({"mpg": [2],
                                                                                    "acceleration": [5],
                                                                                    "has_good_mpg": [True]}))


expected_third_training_set, expected_third_test_set   = (pd.DataFrame({"mpg": [1, 2],
                                    "acceleration": [3, 5],
                                    "has_good_mpg": [False, True]}), pd.DataFrame({"mpg": [3],
                                                                                  "acceleration": [6],
                                                                                  "has_good_mpg": [True]}))

(training_data_1, test_data_1), (training_data_2, test_data_2), (
    training_data_3, test_data_3) = q3.generate_10_fold_cross_validation_data(sample_info)


def test_generate_10_fold_cross_validation_data():
    pdt.assert_frame_equal(training_data_1, expected_first_training_set)
    # pdt.assert_frame_equal(training_data_2, expected_second_training_set)
    # pdt.assert_frame_equal(training_data_3, expected_third_training_set)
    #
    # pdt.assert_frame_equal(test_data_1, expected_first_test_set)
    # pdt.assert_frame_equal(test_data_2, expected_second_test_set)
    # pdt.assert_frame_equal(test_data_3, expected_third_test_set)
