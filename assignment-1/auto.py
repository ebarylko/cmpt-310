import numpy as np
import q3_2 as q32
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
car_statistics = q32.load_auto_data("auto-mpg-regression.tsv")
#-------------------------------------------------------------------------------


# load auto-mpg-regression.tsv, including  Keys are the column names, including mpg.

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are q32.standard and q32.one_hot.

features1 = [('cylinders', q32.standard),
            ('displacement', q32.standard),
            ('horsepower', q32.standard),
            ('weight', q32.standard),
            ('acceleration', q32.standard),
            ('origin', q32.one_hot)]

features2 = [('cylinders', q32.one_hot),
            ('displacement', q32.standard),
            ('horsepower', q32.standard),
            ('weight', q32.standard),
            ('acceleration', q32.standard),
            ('origin', q32.one_hot)]



# Construct the standard data and label arrays
#auto_data[0] has the features for choice features1
#auto_data[1] has the features for choice features2
#The labels for both are the same, and are in auto_values
feature_columns_1, mpg_column = q32.auto_data_and_values(car_statistics, features1)
feature_columns_2, _ = q32.auto_data_and_values(car_statistics, features2)

# #standardize the y-values
scaled_mpg_column, mu, sigma = q32.std_y(mpg_column)

# # #
# #
# # #
# # # #-------------------------------------------------------------------------------
# # # # Analyze auto data
# # # #-------------------------------------------------------------------------------
# # #
# # # #Your code for cross-validation goes here
lambda_vals_1 = np.arange(0, 0.11, 0.01)
lambda_vals_2 = range(0, 210, 20)

feature_columns_1_deg_1, feature_columns_2_deg_1 = map(q32.polynomial_features(1),
                                                       [feature_columns_1, feature_columns_2])

feature_columns_1_deg_2, feature_columns_2_deg_2 = map(q32.polynomial_features(2),
                                                       [feature_columns_1, feature_columns_2])

feature_columns_1_deg_3, feature_columns_2_deg_3 = map(q32.polynomial_features(3),
                                                       [feature_columns_1, feature_columns_2])

calc_rsme_using_frst_lambda_set = tz.partial(q32.avg_rsme_over_lambda,
                                             expected_labels=scaled_mpg_column,
                                             lambda_vals=lambda_vals_1,
                                             num_of_folds=10)

calc_rsme_using_snd_lambda_set = tz.partial(q32.avg_rsme_over_lambda,
                                             expected_labels=scaled_mpg_column,
                                             lambda_vals=lambda_vals_2,
                                             num_of_folds=10)


# print(q32.avg_rsme_over_lambda(feature_columns_1_deg_1, scaled_mpg_column, lambda_vals_1, 10))
# print(q32.avg_rsme_over_lambda(feature_columns_2_deg_1, scaled_mpg_column, lambda_vals_1, 10))
#
#
print(list(tz.mapcat(calc_rsme_using_frst_lambda_set, [feature_columns_1_deg_1, feature_columns_2_deg_1, feature_columns_1_deg_2, feature_columns_2_deg_2])))
print(list(tz.mapcat(calc_rsme_using_snd_lambda_set, [feature_columns_1_deg_3, feature_columns_2_deg_3])))
#
#
# print(q32.avg_rsme_over_lambda(feature_columns_2_deg_2, scaled_mpg_column, lambda_vals_1, 10)[0] * sigma)
# print("The lambda value is ", q32.avg_rsme_over_lambda(feature_columns_2_deg_2, scaled_mpg_column, lambda_vals_1, 10)[1])