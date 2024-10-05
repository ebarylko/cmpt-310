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