import q2 as q
import pandas as pd
import pandas.testing as pdt


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
    pdt.assert_frame_equal(q.scale_features(sample), expected_scaled_data)


sample_info = pd.DataFrame({"mpg": [1, 2, 3],
                            "acceleration": [3, 5, 6],
                            "has_good_mpg": [False, True, True]})

expected_first_training_set, expected_first_test_set = (pd.DataFrame({"mpg": [2, 3],
                                                                      "acceleration": [5, 6],
                                                                      "has_good_mpg": [True, True]},
                                                                     index=[1, 2]),
                                                        pd.DataFrame({"mpg": [1],
                                                                      "acceleration": [3],
                                                                      "has_good_mpg": [False]}))

expected_second_training_set, expected_second_test_set = (pd.DataFrame({"mpg": [1, 3],
                                                                         "acceleration": [3, 6],
                                                                         "has_good_mpg": [False, True]}),
                                                          pd.DataFrame({"mpg": [2],
                                                                        "acceleration": [5],
                                                                        "has_good_mpg": [True]}))


expected_third_training_set, expected_third_test_set = (pd.DataFrame({"mpg": [1, 2],
                                                                        "acceleration": [3, 5],
                                                                        "has_good_mpg": [False, True]}),
                                                        pd.DataFrame({"mpg": [3],
                                                                      "acceleration": [6],
                                                                      "has_good_mpg": [True]}))


actual_data_1, actual_data_2, actual_data_3 = q.generate_10_fold_cross_validation_data(sample_info)


expected_1 = (pd.DataFrame({"mpg": [2, 3],
                            "acceleration": [5, 6],
                            "has_good_mpg": [True, True]},
                           index=[1, 2]),
              pd.DataFrame({"mpg": [1],
                            "acceleration": [3],
                            "has_good_mpg": [False]}))

expected_2 = (pd.DataFrame({"mpg": [1, 3],
                            "acceleration": [3, 6],
                            "has_good_mpg": [False, True]},
                           index=[0, 2]),
              pd.DataFrame({"mpg": [2],
                            "acceleration": [5],
                            "has_good_mpg": [True]},
                           index=[1]))

expected_3 = (pd.DataFrame({"mpg": [1, 2],
                            "acceleration": [3, 5],
                            "has_good_mpg": [False, True]}),
              pd.DataFrame({"mpg": [3],
                            "acceleration": [6],
                            "has_good_mpg": [True]},
                           index=[2]))


def assert_training_and_test_data_match(expected, actual):
    expected_training_data, expected_test_data = expected
    actual_training_data, actual_test_data = actual
    pdt.assert_frame_equal(expected_training_data, actual_training_data)
    pdt.assert_frame_equal(expected_test_data, actual_test_data)


def test_generate_10_fold_cross_validation_data():
    assert_training_and_test_data_match(expected_1, actual_data_1)
    assert_training_and_test_data_match(expected_2, actual_data_2)
    assert_training_and_test_data_match(expected_3, actual_data_3)
