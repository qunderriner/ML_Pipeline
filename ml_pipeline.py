"""
Simple pipeline for ML projects
Author: Quinn Underriner
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import seaborn as sns
sns.set()


def load_data(filename):
    """
    This function loads the dataset from a CSV 
    """
    df = pd.read_csv(filename)
    return df

"""
functions for exploration 
"""
def per_zip(df, col_name, zip_col_name):
    """
    Returns the averages a each given column per zip code 
    Inputs:
        df: dataframe
        col_name (str): column name you want to target
        zip_col_name (str): name of column in dataframe with zipcode info 
    """
    df_new = df[[col_name, zip_col_name]]
    per_z = df_new.groupby((zip_col_name)).mean().reset_index()
    per_z = per_z.round(2)
    per_z = per_z.rename(index=str, columns={0: "Count of " + col_name})
    return per_z 

def percentage_calc(df, col_name, value):
    """
    Prints percent of data over a certain threshold. 
    inputs:
        df: dataframe 
        col_name (str): column to do this for. 
        value (int): threshold to go over 
    """
    print(len(df[df[col_name]>value]) / len(df))

def find_corr(df):
    """
    Generates a heatmap of correlations 
    Inputs:
        df: dataframe 
    """
    corr = df.corr()
    sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns)

"""
functions for preprocessing
"""

def impute(df, mean_cols=None, median_cols=None):
    """
    This function takes null values in a column and fills it either with the median value
    for the column or the mean, depending on the input
    inputs 
    """
    for col in mean_cols:
        df[col] = df[col].fillna(df[col].mean())
    for col in median_cols:
        df[col] = df[col].fillna(df[col].median())
    return df


"""
feature generation
"""


def discretize(df, colname):
    """
    This function discretizes a continuous variable (using quartiles)
    Inputs:
        df (dataframe)
        colname (str) name of column to discretize 
    """
    df[colname] = pd.qcut(df[colname], 4, labels=[1, 2, 3, 4])
    return df


def dummy(df, colname):
    """
    Takes a categorical variable and creates binary/dummy variables
    Inputs:
        df (dataframe)
        colname (str) name of column to make dummys  
    """
    dummies = pd.get_dummies(df[colname]).rename(columns=lambda x: colname + "_" + str(x))
    df = pd.concat([df, dummies], axis=1)
    df = df.drop([colname], axis=1)
    return df


def get_xy(df, response, features):

    """
    Create data arrays for the X and Y values needed to be plugged into the model
    Inputs:
        df (dataframe) - the dataframe 
        response (str - the y value for the model 
        features (list of strings) - the x values for the model 
    """
    y = df[response].to_numpy()
    X = df[features].to_numpy()
    return X, y


def classify_lgreg(X, y):
    """
    Builds and returns trained logistic regression model
    Inputs:
        X (array) - Features for the model
        y (array) - What is being predicted

    """
    lg = LogisticRegression()
    lg.fit(X, y)
    return lg


def accuracy(true_values, predicted_values):
    """
    Computes the accuracy of the prediction
    Inputs:
        true_values (array) actual predicted values from data set
        predicted_values (array) what our model predicted 
    """

    return np.mean(true_values == predicted_values)
