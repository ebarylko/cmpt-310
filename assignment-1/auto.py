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

# car_statistics = pd.read_csv('auto-mpg-regression.csv').filter(items=wanted_features)
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
auto_data = [0, 0]
auto_values = 0
auto_data[0], auto_values = q3.auto_data_and_values(car_statistics, features1)
auto_data[1], _ = q3.auto_data_and_values(car_statistics, features2)
#
# #standardize the y-values
auto_values, mu, sigma = q3.std_y(auto_values)
#
# #-------------------------------------------------------------------------------
# # Analyze auto data
# #-------------------------------------------------------------------------------
#
# #Your code for cross-validation goes here