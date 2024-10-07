import q3
import q3 as q3
import numpy as np

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
car_statistics = q3.load_auto_data("auto-mpg-regression.tsv")
#-------------------------------------------------------------------------------


# load auto-mpg-regression.tsv, including  Keys are the column names, including mpg.

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are q3.standard and q3.one_hot.

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



# Construct the standard data and label arrays
#auto_data[0] has the features for choice features1
#auto_data[1] has the features for choice features2
#The labels for both are the same, and are in auto_values
feature_columns_1, mpg_column = q3.auto_data_and_values(car_statistics, features1)
feature_columns_2, _ = q3.auto_data_and_values(car_statistics, features2)

# #standardize the y-values
scaled_mpg_column, mu, sigma = q3.std_y(mpg_column)

lambda_vals_1 = np.arange(0, 0.11, 0.01)
lambda_vals_2 = range(0, 210, 20)

feature_columns_1_deg_1, feature_columns_2_deg_1 = map(q3.polynomial_features(1),
                                                       [feature_columns_1, feature_columns_2])

feature_columns_1_deg_2, feature_columns_2_deg_2 = map(q3.polynomial_features(2),
                                                       [feature_columns_1, feature_columns_2])

feature_columns_1_deg_3, feature_columns_2_deg_3 = map(q3.polynomial_features(3),
                                                       [feature_columns_1, feature_columns_2])

print(q3.avg_rsme_over_lambda(feature_columns_1_deg_1, mpg_column, lambda_vals_1, 10))
print(q3.avg_rsme_over_lambda(feature_columns_2_deg_1, mpg_column, lambda_vals_1, 10))


print(q3.avg_rsme_over_lambda(feature_columns_1_deg_2, mpg_column, lambda_vals_1, 10))
print(q3.avg_rsme_over_lambda(feature_columns_2_deg_2, mpg_column, lambda_vals_1, 10))

print(q3.avg_rsme_over_lambda(feature_columns_1_deg_3, mpg_column, lambda_vals_2, 10))
print(q3.avg_rsme_over_lambda(feature_columns_2_deg_3, mpg_column, lambda_vals_2, 10))


#
# #-------------------------------------------------------------------------------
# # Analyze auto data
# #-------------------------------------------------------------------------------
#
# #Your code for cross-validation goes here
