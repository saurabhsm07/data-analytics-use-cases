# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:16:21 2019

@author: saurabh_mahambrey
"""

# required libraries:
import pandas as pd;
import numpy as np;


# importing training and test data into pandas dataframe
def import_data(path):
    return pd.read_csv(path)


# utility functions
def print_info(df):
    return df.info()


def print_description(df):
    return df.describe()


def print_head(df, count=5):
    return df.head(count)


# function returns a dictionary of percentage of null values in a dataframe
def get_null_percent(df):
    null_percent_dict = {}

    for col in df.columns:
        null_percent_dict[col] = df[df[col].isnull() == True].loc[:, 'Survived'].count() / df.shape[0] * 100

    return null_percent_dict


# get list of attributes with atleast 1 cell having a null value
def get_attributes_with_nulls(df):
    df_nulls = pd.Series(df.isna().any())
    return df_nulls[df_nulls == True].index.tolist()


def get_attributes_with_nulls_inorder(df):
    return (df.isnull()
            .sum(axis=0)
            .loc[df.isnull().sum() > 0]
            .sort_values(ascending=True))
