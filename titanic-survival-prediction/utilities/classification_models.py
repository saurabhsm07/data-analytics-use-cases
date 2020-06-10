# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:44:49 2019

@author: saurabh_mahambrey
"""


# Classification Models :

# Logistic Regression

def logistic_regression_model(df):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'Survived'], df['Survived'],
                                                        test_size=0.25, random_state=0)
    clf = LogisticRegression(solver='liblinear')  # solver : liblinear good for small datasets
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


# Grid Searched : Logistic regression

def gridSearch_logistic_regression_model(df):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    logistic_reg = LogisticRegression(solver='liblinear')
    parameters = {'penalty': ('l1', 'l2'),
                  'C': [0.1, 10]
                  }
    clf = GridSearchCV(logistic_reg, parameters, cv=5, iid=True)

    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'Survived'], df['Survived'],
                                                        test_size=0.25, random_state=0)
    X_train, X_test = min_max_scale_data(X_train, X_test)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


# Scaled: Logistic Regression

def scaled_logistic_regression(df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    clf = LogisticRegression(solver='liblinear')
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'Survived'], df['Survived'],
                                                        test_size=0.25, random_state=0)
    X_train, X_test = min_max_scale_data(X_train, X_test)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)
