import q3 as q
import pandas as pd
import operator as op
import functools as ft

sample_data = pd.DataFrame({"mpg": [1, 2, 3, 4],
                            "displacement": [3, 4, 5, 6]})

feature_to_scaling_function = {"mpg": ft.partial(op.add, 1),
                               "displacement": ft.partial(op.mul, 2)}

expected = pd.DataFrame({"mpg": [2, 3, 4, 5],
                         "displacement": [6, 8, 10, 12]})


def test_normalize_and_one_hot_encode_data():
    assert q.normalize_and_one_hot_encode_data(sample_data, feature_to_scaling_function) ==  expected