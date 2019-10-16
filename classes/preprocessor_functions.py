# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:29:19 2019

@author: saurabh_mahambrey
"""
#required libraries:
import pandas as pd;    
import numpy as np;

def numeric_mapper(df, column_name):
    mapper = {}
    data_list = df[column_name].unique()
    data_list = ['missing' if pd.isnull(x) else x for x in data_list]
    data_list.sort()
    for i in range(0, len(data_list)):
        if data_list[i] == 'missing':
            mapper[np.nan] = 404
        else:    
            mapper[data_list[i]] = i
    return mapper
    
def data_preprocessor(df):
 
    df = (df.rename({'SibSp' : '# of Siblings', 
                                             'Parch': '# of Parents', 
                                             'Sex' : 'Gender',
                                             'Pclass' : 'Class'},
                                            axis = 1)
                                    .drop(['Name', 'Ticket', 'PassengerId'], axis = 1)
                                    .astype({'Gender' : pd.api.types.CategoricalDtype(df['Sex'].unique(), ordered=False), 
                                             'Class' : pd.api.types.CategoricalDtype(df['Pclass'].unique(), ordered=True)})
#                                     .replace({'Embarked' : {np.NaN : 'un-known'}})
                                    .replace({'Gender' : numeric_mapper(df, 'Sex'),
                                              'Embarked' : numeric_mapper(df, 'Embarked'),
                                              'Cabin' : numeric_mapper(df, 'Cabin'),
                                              'Age' : {np.nan : 404}})
                                    
    #                                 .loc[:]
                      )
    return df

#### MinMax Scaler :
    
def min_max_scale_data(train_set, test_set):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range= (-1,1))
    train_scaled = scaler.fit_transform(train_set)
    test_scaled = scaler.transform(test_set)
    return (train_scaled, test_scaled)




#### Standard Scaler:

# scaled data to a mean = 0 , std = 1
def standard_scale_data(train_set, test_set):
    from sklearn.preprocessing import StandardScaler
    std_scaler = StandardScaler()
    std_scaler.fit(train_set)
    train_scaled = std_scaler.transform(train_set)
    test_scaled = std_scaler.transform(test_set)
    return (train_scaled, test_scaled)

#### PCA (Principal Component Analysis) : 
    
def get_pca_dataset(train_set, test_set, features = 2):
    from sklearn.decomposition import PCA
    train_scaled, test_scaled = standard_scale_data(train_set, test_set)
    pca = PCA(n_components= features)
    pca.fit(train_scaled)
    pca_train = pca.transform(train_scaled)
    pca_test = pca.transform(test_scaled)
    return (pca_train, pca_test)

#### Train Test Splitter:
    
def train_test_split_data(df):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns!= 'Survived'], df['Survived'], test_size = 0.25, random_state = 0)
    return (X_train, X_test, y_train, y_test)