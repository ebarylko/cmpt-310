import numpy as np
import q3 as q3
import toolz as tz

#-------------------------------------------------------------------------------
# Auto Data
# wanted_features = ["cylinders",
#                    "displacement",
#                    "horsepower",
#                    "weight",
#                    "acceleration",
#                    "origin",
#                    "mpg"]
#
# car_statistics2 = pd.read_csv('auto-mpg-regression.csv').filter(items=wanted_features)
auto_data_all = q3.load_auto_data("auto-mpg-regression.tsv")
#-------------------------------------------------------------------------------


# load auto-mpg-regression.tsv, including  Keys are the column names, including mpg.

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are q32.standard and q32.one_hot.

features1 = [('cylinders', q3.standard),
            ('displacement', q3.standard),
            ('horsepower', q3.standard),
            ('weight', q3.standard),
            ('acceleration', q3.standard),
            ('origin', q3.one_hot)]

features2 = [('cylinders', q3.one_hot),
            ('displacement', q3.standard),
            ('horsepower', q3.standard),
            ('weight', q3.standard),
            ('acceleration', q3.standard),
            ('origin', q3.one_hot)]



# # Construct the standard data and label arrays
# #auto_data[0] has the features for choice features1
# #auto_data[1] has the features for choice features2
# #The labels for both are the same, and are in auto_values
# feature_columns_1, mpg_column = q32.auto_data_and_values(car_statistics, features1)
# feature_columns_2, _ = q32.auto_data_and_values(car_statistics, features2)
#
# # #standardize the y-values
# scaled_mpg_column, mu, sigma = q32.std_y(mpg_column)
#
# # # #
# # #
# # # #
# # # # #-------------------------------------------------------------------------------
# # # # # Analyze auto data
# # # # #-------------------------------------------------------------------------------
# # # #
# # # # #Your code for cross-validation goes here
# lambda_vals_1 = np.arange(0, 0.11, 0.01)
# lambda_vals_2 = range(0, 210, 20)
#
# feature_columns_1_deg_1, feature_columns_2_deg_1 = map(q32.polynomial_features(1),
#                                                        [feature_columns_1, feature_columns_2])
#
# feature_columns_1_deg_2, feature_columns_2_deg_2 = map(q32.polynomial_features(2),
#                                                        [feature_columns_1, feature_columns_2])
#
# feature_columns_1_deg_3, feature_columns_2_deg_3 = map(q32.polynomial_features(3),
#                                                        [feature_columns_1, feature_columns_2])
#
# calc_rsme_using_frst_lambda_set = tz.partial(q32.avg_rsme_over_lambda,
#                                              expected_labels=scaled_mpg_column,
#                                              lambda_vals=lambda_vals_1,
#                                              num_of_folds=10)
#
# calc_rsme_using_snd_lambda_set = tz.partial(q32.avg_rsme_over_lambda,
#                                              expected_labels=scaled_mpg_column,
#                                              lambda_vals=lambda_vals_2,
#                                              num_of_folds=10)
#
#
# # print(q32.avg_rsme_over_lambda(feature_columns_1_deg_1, scaled_mpg_column, lambda_vals_1, 10))
# # print(q32.avg_rsme_over_lambda(feature_columns_2_deg_1, scaled_mpg_column, lambda_vals_1, 10))
# #
# #
# print(list(tz.mapcat(calc_rsme_using_frst_lambda_set, [feature_columns_1_deg_1, feature_columns_2_deg_1, feature_columns_1_deg_2, feature_columns_2_deg_2])))
# print(list(tz.mapcat(calc_rsme_using_snd_lambda_set, [feature_columns_1_deg_3, feature_columns_2_deg_3])))
# #
# #
# # print(q32.avg_rsme_over_lambda(feature_columns_2_deg_2, scaled_mpg_column, lambda_vals_1, 10)[0] * sigma)
# # print("The lambda value is ", q32.avg_rsme_over_lambda(feature_columns_2_deg_2, scaled_mpg_column, lambda_vals_1, 10)[1])

auto_data = [0, 0]

auto_values = 0

FEATURE = 2

if FEATURE == 1:
    print("features1")
    auto_data[0], auto_values = q3.auto_data_and_values(auto_data_all, features1)
    auto_data[1], _ = q3.auto_data_and_values(auto_data_all, features2)
else:
    print("features2")
    auto_data[0], auto_values = q3.auto_data_and_values(auto_data_all, features2)
    auto_data[1], _ = q3.auto_data_and_values(auto_data_all, features1)

#standardize the y-values

auto_values, mu, sigma = q3.std_y(auto_values)

#------------------------------------------------------------------------

# Analyze auto data

#------------------------------------------------------------------------

# polynomial order 1 and 2

reg_params_1 = [i* 0.01 for i in range(0, 11)]

print("order 1")
for reg_param in reg_params_1:
    rmse = q3.xval_learning_alg(q3.make_polynomial_feature_fun(1)(auto_data[0]), auto_values, reg_param, 10)
    print("reg_param: ", reg_param, " RMSE: ", rmse*sigma)

print("order 2")

for reg_param in reg_params_1:
    rmse = q3.xval_learning_alg(q3.make_polynomial_feature_fun(2)(auto_data[0]), auto_values, reg_param, 10)
    print("reg_param: ", reg_param, " RMSE: ", rmse*sigma)


# polynomial order 3

reg_params_2 = [i* 20 for i in range(0, 11)]

print("order 3")

for reg_param in reg_params_2:
    rmse = q3.xval_learning_alg(q3.make_polynomial_feature_fun(3)(auto_data[0]), auto_values, reg_param, 10)
    print("reg_param: ", reg_param, " RMSE: ", rmse*sigma)