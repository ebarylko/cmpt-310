import numpy as np
import q3 as q3
import pandas as pd

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
#
# #-------------------------------------------------------------------------------
# # Analyze auto data
# #-------------------------------------------------------------------------------
#
# #Your code for cross-validation goes here
wanted_features = ["cylinders",
                   "displacement",
                   "horsepower",
                   "weight",
                   "acceleration",
                   "mpg"]
car_statistics2 = (q3.read_car_data('auto-mpg-regression.tsv')
                   .pipe(q3.filter_features, wanted_features)
                   .pipe(q3.add_mileage_label)
                   .drop('mpg', axis="columns")
                   #.pipe(q3.scale_features)
                   )

print(car_statistics2)

