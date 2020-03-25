# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:42:09 2019

@author: saurabh_mahambrey
"""
### Visualizations :

#### Heatmap : correlation matrix visualization :


# plot a heatmap of correlation matrix of attribute variables
def plot_correlation_heatmap(correlation_data):
#    %matplotlib inline
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 6))
    import seaborn as sns
    return sns.heatmap(correlation_data, annot = True)

#### PCA : 2 variable scatter plot visualization 
#### <font color= 'red'>  IN Progress : create 2 sub plots 1: 2 variable scatter using pca and 2: 2 variable scatter based on correlation </font>
    
def plot_pca_scatterplot(train, test):
#    %matplotlib inline
    import matplotlib.pyplot as plt
    plt.scatter(train[:,0], train[:,1], color = 'green', marker = 'X')
    plt.scatter(test[:, 0], test[:, 1], color = 'red', marker = 'X')