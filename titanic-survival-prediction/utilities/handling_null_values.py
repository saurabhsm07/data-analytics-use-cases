# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:31:44 2019

@author: saurabh_mahambrey
"""
# required libraries:
import pandas as pd;
import numpy as np;


# Dealing With Null Values In Attributes :

# CASE 1 : Removing attributes with null values:

# In training only 2 attributes contains a major chunk of null data : Age and Cabin
def drop_null_attributes(df):
    df = (df.dropna(axis='columns')
          )
    return df


# CASE 2: Remove rows with null values:
def drop_null_rows(df):
    df = (df.dropna(axis='rows')
          )
    return df


# CASE 3: Replacing null attribute values with mean, median or mode

# replace nulls with averages 
def replace_null_with_mean(col):
    if col.dtype.name == "category":
        col = col.replace(np.nan, col.mode())
    else:
        col = col.replace(np.nan, col.mean())
    return col


def replace_nulls_phase_1(df):
    df = (df.apply(replace_null_with_mean, axis=0)
          )
    return df


"""
CASE 4: Group attributes with similar values and replace null values with mean or mode values of that specific group
        Function to return attributes with greatest correlation with the provided attribute
"""


def top_correlations(df, attribute, count=2):
    correlations_df = df.corr()
    correlation_attribute = correlations_df[attribute]
    correlation_attribute = correlation_attribute.to_frame()
    correlation_attribute[attribute + '_mod'] = [x if x > 0 else -1 * x for x in correlations_df[attribute]]
    return (correlation_attribute.sort_values(attribute + '_mod', ascending=False)[1: (count + 1)].loc[:, attribute])


"""
form a dictionary structure of column with nulls as keys and attributes they are most correlated to as a list of values
"""


def get_top_corr_dict(attribute_with_nulls, titanic_train):
    top_corr_dict = {}
    for attribute in attribute_with_nulls:
        top_corr_dict[attribute] = top_correlations(titanic_train, attribute)
    return top_corr_dict


"""
dataframe of values based on averages columns with null values based on groupby result of previous step
"""


def get_vals_based_on_corr_groups(titanic_train, top_corr_dict, attr_type):
    values_based_corr_dict = {}
    for attr in top_corr_dict.keys():
        if attr_type[attr] == 'category':
            df = titanic_train.groupby(top_corr_dict[attr].index.tolist(), as_index=False)[attr].agg(
                lambda x: x.value_counts().index[0] if len(x.value_counts()) is not 0 else
                titanic_train[attr].value_counts().index[1])
        else:
            df = titanic_train.groupby(top_corr_dict[attr].index.tolist(), as_index=False)[attr].agg(
                lambda x: x.mean() if pd.isnull(x.mean()) is not True else titanic_train[attr].value_counts().index[0])
        values_based_corr_dict[attr] = df
    return values_based_corr_dict


"""
CASE 5 A: replace null values by creating a simple prediction model from remaining non null attributes in  (PARALLEL)

[x for x in titanic_train_5.columns if x not in [y for y in attribute_with_nulls if y != 'Age']] - 
list comprehension to remove non important nulls

split groups into train and predict dataframe sets based on condition if attribute.val = np.nan
"""


def attr_train_predict_split(df, attribute_with_nulls, attr):
    df = df.loc[:, [x for x in df.columns if x not in [y for y in attribute_with_nulls if y != attr]]]
    train_set = df[np.isnan(df[attr]) == False]
    predict_set = df[np.isnan(df[attr])]
    return (train_set, predict_set)


"""
predict substitutes for null values by training a simple KNN model on the data split of non null attribute values
"""


def predict_nulls(train_set, predict_set, attr, attr_type):
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    X_train = train_set.loc[:, train_set.columns != attr]
    y_train = train_set[attr]

    X_predict = predict_set.loc[:, predict_set.columns != attr]
    y_predict = predict_set[attr]

    if (attr_type == 'category'):
        return KNeighborsClassifier().fit(X_train, y_train).predict(X_predict)
    else:
        return KNeighborsRegressor().fit(X_train, y_train).predict(X_predict)


"""
CASE 5 B: replace null values by creating a simpe prediction model from remaining non null attributes in (SERIES)

split groups into train and predict dataframe sets based on condition if attribute.val = np.nan
"""


def attr_train_predict_split_inorder(df, attribute_with_nulls_inorder, attr):
    df = df.loc[:, [x for x in df.columns if (df[x].isna().any() != True) | (x == attr)]]
    train_set = df[np.isnan(df[attr]) == False]
    predict_set = df[np.isnan(df[attr])]
    return (train_set, predict_set)


"""
predict substitutes for null values by training a simple KNN model on the data split of non null attribute values
"""


def predict_nulls_inorder(train_set, predict_set, attr, attr_type):
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    X_train = train_set.loc[:, train_set.columns != attr]
    y_train = train_set[attr]

    X_predict = predict_set.loc[:, predict_set.columns != attr]
    y_predict = predict_set[attr]

    if (attr_type == 'category'):
        return KNeighborsClassifier().fit(X_train, y_train).predict(X_predict)
    else:
        return KNeighborsRegressor().fit(X_train, y_train).predict(X_predict)
