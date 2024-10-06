import q3 as q
import numpy as np
import numpy.testing as npt

# sample_data = pd.DataFrame({"mpg": [1, 2, 3, 4],
#                             "displacement": [3, 4, 5, 6]})
#
# feature_to_scaling_function = [("mpg", ft.partial(op.add, 1)),
#                                ("displacement", ft.partial(op.mul, 2))]
#
# expected_df = pd.DataFrame({"mpg": [2, 3, 4, 5],
#                          "displacement": [6, 8, 10, 12]})
#
#
# def test_normalize_and_one_hot_encode_data():
#     pdt.assert_frame_equal(q.normalize_and_one_hot_encode_data(sample_data, feature_to_scaling_function), expected_df)
#
#
# sample = pd.Series([1, 2, 3, 3])
#
# expected_encoding = pd.DataFrame([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], columns=[1, 2, 3])
#
#
# def test_one_hot():
#     pdt.assert_frame_equal(expected_encoding, q.one_hot(sample), check_dtype=False)


def test_mean_square_loss_th():
    X = np.array([[ 1.,  2.,  3.,  4.], [ 1.,  1.,  1.,  1.]])
    Y = np.array([[ 1. ,  2.2,  2.8,  4.1]])
    th = np.array([[ 1.  ], [ 0.05]]) ; th0 = np.array([[ 2.]])
    npt.assert_array_equal(q.d_mean_square_loss_th(X[:,0:1], Y[:,0:1], th, th0), np.array([[4.1], [4.1]]))

def test_d_mean_square_loss_th0():
    assert q.d_mean_square_loss_th0(np.array([[ 1.,  2.,  3.,  4.], [ 1.,  1.,  1.,  1.]]),
                                    np.array([[1., 2.2, 2.8, 4.1]]),
                                    np.array([[1.], [0.05]]),
                                    np.array([[2.]])) == 4.05


def test_d_ridge_obj_th():
    X = np.array([[1., 2., 3., 4.], [1., 1., 1., 1.]])
    Y = np.array([[1., 2.2, 2.8, 4.1]])
    th = np.array([[1.], [0.05]]);
    th0 = np.array([[2.]])
    npt.assert_array_equal(q.d_ridge_obj_th(X, Y, th, th0, 0.0), np.array([[10.15], [4.05]]))
    npt.assert_array_equal(q.d_ridge_obj_th(X, Y, th, th0, 0.5), np.array([[11.15], [4.1]]))
    npt.assert_array_equal(q.d_ridge_obj_th(X, Y, th, th0, 100.), np.array([[210.15], [14.05]]))


def test_d_ridge_obj_th0():
    X = np.array([[1., 2., 3., 4.], [1., 1., 1., 1.]])
    Y = np.array([[1., 2.2, 2.8, 4.1]])
    th = np.array([[1.], [0.05]]);
    th0 = np.array([[2.]])
    npt.assert_array_equal(q.d_ridge_obj_th0(X, Y, th, th0, 0.0), np.array([[4.05]]))
    npt.assert_array_equal(q.d_ridge_obj_th0(X, Y, th, th0, 0.5), np.array([[4.05]]))
    npt.assert_array_equal(q.d_ridge_obj_th0(X, Y, th, th0, 100.), np.array([[4.05]]))


