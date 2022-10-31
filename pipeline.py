# Pipeline
# Author: Linh Dinh


import sqlite3
import os
import csv
import json
import requests
import re
import datetime
from datetime import timedelta
import time
import importlib

import pipeline
import process_data

import scipy
from scipy import stats
import pandas as pd
import numpy as np

from geopandas import GeoDataFrame
import geopandas as gpd

import json
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from matplotlib import colors
from matplotlib.patches import RegularPolygon

import seaborn as sns

import shapely
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely import wkt

import plotly.express as px
import plotly.graph_objects as go

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains 
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

import sys
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder


### Read Data
def read_data(filename):
    '''
    Purpose: Load data (currently only supports csv files)
    
    Inputs:
        filename (str): the filename of your data
        DATE_COL (str): if your data has a DateTime 
        variable, you can pass its name and the function
        will convert the variable from str to datetime.

    Returns: a pd.dataframe of the data'''

    #if not os.path.exists(filename):
        #return None

    df = pd.read_csv(filename)

    return df


### Clean Data
def convert_to_str(df, COLS):
    '''Convert vars to strings'''

    for c in COLS:
        if c in df.columns:
            if df[c].dtype != 'O':
                df[c] = df[c].astype(str)

    return None


def convert_to_flt(df, COLS):
    '''Convert vars to strings'''

    for c in COLS:
        if c in df.columns:
            if df[c].dtype != 'float64':
                df[c] = df[c].astype(float)

    return None


def convert_to_int(df, COLS):
    '''Convert vars to strings'''

    for c in COLS:
        if c in df.columns:
            if df[c].dtype != 'int64':
                df[c] = df[c].astype(int)

    return None


def convert_to_timeseries(df, DATE_COLS=None):
    '''Convert timeseries variables'''

    if DATE_COLS:
        for d in DATE_COLS:
            df[d] = pd.to_datetime(df[d])

    return None


def create_timeseries_features(df, date_feature, hourly=True, 
    daily=True, monthly=True, yearly=True):
    '''
    Purpose: create some standard timeseries variables from the Date variable
    in the original data
    Inputs:
        df (dataframe): data to explore
        date_feature: Date variable in the original data

    Returns: modify the existing df with additional timeseries variables
    '''
    
    if hourly:
        df['Hourly'] = df[date_feature].dt.hour
    if daily: 
        df['Daily'] = df[date_feature].dt.date
    if monthly:
        df['Monthly'] = df[date_feature].dt.month
        df['MonthYear'] = '01-' + df['Monthly'].astype(str)+ '-' + df[date_feature].dt.year.astype(str)
        #df['MonthYear'] = df[date_feature].dt.to_period('M').astype(str)
        df['MonthYear'] = pd.to_datetime(df['MonthYear'], format='%d-%m-%Y').dt.date
    if yearly:
        df['Yearly'] = df[date_feature].dt.year

    return None


def haversine(coord1, coord2):
    # Coordinates in decimal degrees (e.g. 43.60, -79.49)
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    R = 6371000  # radius of Earth in meters
    phi_1 = np.radians(lat1)
    phi_2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a),np.sqrt(1 - a))
    meters = R * c  # output distance in meters
    km = meters / 1000.0  # output distance in kilometers
    meters = round(meters)
    km = round(km, 3)
    miles = round(meters * 0.00062137)
    print(f"Distance: {meters} m")
    print(f"Distance: {km} km")
    print(f"Distance: {miles} miles")
    
    return miles


def plot_scatter(df, xvar, yvar, title=None, hue=None, figsize=(30,15)):
    if not hue:
        fig, ax1 = plt.subplots(figsize=figsize)
        g1 = sns.regplot(x=xvar, y=yvar, data=df, ax=ax1)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=g1.get_lines()[0].get_xdata(), y=g1.get_lines()[0].get_ydata()) 
        g1.set_xlabel(xvar)
        g1.set_ylabel(yvar)     
        g1.set_xlim(min(df[xvar].min(), df[yvar].min()), max(df[xvar].max(), df[yvar].max()))
        g1.set_ylim(min(df[xvar].min(), df[yvar].min()), max(df[xvar].max(), df[yvar].max()))
    
    else:
        g1 = sns.lmplot(x=xvar, y=yvar, data=df, hue=hue, height=figsize[0])
        g1 = (g1.set_axis_labels([xvar, yvar])
             .set(xlim=(min(df[xvar].min(), df[yvar].min()), max(df[xvar].max(), df[yvar].max())),
                  ylim=(min(df[xvar].min(), df[yvar].min()), max(df[xvar].max(), df[yvar].max()))))
    
    corr = round(df[xvar].corr(df[yvar]),3)
    
    line_xy = [min(df[xvar].min(), df[yvar].min()), max(df[xvar].max(), df[yvar].max())]
    
    g2 = sns.lineplot(x=line_xy, y=line_xy, color='black')
    
    if not hue:
        g2.legend(["Line: y = " + str(round(intercept,1)) + " + " + str(round(slope,1)) + " * x"], loc='lower right')      
    
    g2.set_xlabel(xvar)
    g2.set_ylabel(yvar) 
    g2.set(title=title + " (Corr: " + str(corr) + ")")

    if not hue:
        return fig

    return g1


def plot_2_axis(df, timevar, features, ylabel_1, ylabel_2, title=None, figsize=(30,15)):
    fig, ax1 = plt.subplots(figsize=figsize)

    for v1 in features['ax1']:
        sns.lineplot(x=timevar, y=v1[0], data=df, ax=ax1, color=v1[1])
    
    for i, l in enumerate(ax1.lines): 
        l.set_linestyle(features['ax1'][i][2])
        
    ax1.set_ylabel(ylabel_1)
    ax1.legend(features['ax1'], loc='upper left')   
    ax1.set(title=title)
    
    if features['ax2']:
        ax2 = ax1.twinx()
        for v2 in features['ax2']:
            sns.lineplot(x=timevar, y=v2[0], data=df, ax=ax2, color=v2[1])

        ax2.set_ylabel(ylabel_2)
        for ii, ll in enumerate(ax2.lines): 
            ll.set_linestyle(features['ax2'][ii][2])
        ax2.legend(features['ax2'], loc='upper right')
    
    fig.autofmt_xdate()

    return fig   


def remove_outliers(df, vars, pri=False):
    if pri:
        print("NAs count", df[vars][df[vars].isnull().all(1)].shape[0], "(", round(df[vars][df[vars].isnull().all(1)].shape[0]*100/df.shape[0], 3), "% of the original dataframe )")
        print("Before dropping NAs", df.shape)
    nonan = df.dropna(subset=vars)
    if pri:
        print("After dropping NAs", nonan.shape)
    z = np.abs(stats.zscore(nonan[vars]))
    if pri:
        print("Outliers", nonan.shape[0] - nonan[(z<4).all(axis=1)].shape[0], "(", round((1 - nonan[(z<4).all(axis=1)].shape[0]/nonan.shape[0])*100, 3), "% of the remaining dataframe )")
    df_limit = nonan[(z<4).all(axis=1)]
    if pri:
        print("After removing outliers", df_limit.shape, "(", round(df_limit.shape[0]*100/df.shape[0], 3), "% of the orgiginal dataframe )" )
    
    return df_limit


def original_vs_removed_outliers_plot(org, rmo, vars, timeseries, figsize=(30,15)):
    fig, axs = plt.subplots(len(vars), 4, figsize=figsize)
    for i, v in enumerate(vars):
        plot_features_hist(org, v, title = "Original: Hist of " + v, ax=axs[i][0])
        plot_features_hist(rmo, v, title = "Removed Outliers: Hist of " + v, ax=axs[i][1])
        plot_feature_timeseries(org, timeseries, v, title = "Original: Timeseries of " + v, ax=axs[i][2])
        plot_feature_timeseries(rmo, timeseries, v, title = "Removed Outliers: Timeseries of " + v, ax=axs[i][3])
        
    return fig


def create_summary_timeseries_dataset(df, time_features, cat_features, sum_features, max_features, all_feature, print=False):
    
    assert (all_feature and not sum_features and not max_features) or (not all_feature and (sum_features or max_features))
    if not all_feature:
        df_sum = df[time_features + cat_features + sum_features].groupby(time_features + cat_features).sum().reset_index()
        df_max = df[time_features + cat_features + max_features].groupby(time_features + cat_features).max().reset_index()
        df_overall = df_sum.merge(df_max, on=time_features + cat_features, how='inner')
        df_overall['mean'] = df_overall[sum_features[0]] / df_overall[sum_features[1]]
        if time_features:
            convert_to_timeseries(df_overall, time_features)
        
    if all_feature:
        df_overall = df[time_features + cat_features + all_feature].groupby(
        time_features + cat_features).agg(
        [('median', np.median), ('mean', np.mean), ('max', np.max), ('sum', np.sum), ('count', lambda x: x.shape[0])]).reset_index()
        if time_features:
            convert_to_timeseries(df_overall, time_features)
        
        df_overall.columns = df_overall.columns.to_flat_index()
        df_overall = df_overall.rename(columns={df_overall.columns[-5]: "median",
                                                  df_overall.columns[-4]: "mean", df_overall.columns[-3]: "max",
                                                  df_overall.columns[-2]: "sum", df_overall.columns[-1]: "count"})
    
    for i, v in enumerate(time_features + cat_features):
        df_overall = df_overall.rename(columns={df_overall.columns[i]: v})

    if print:
        print("Original dataframe:", df.shape)
        print("New dataframe:", df_overall.shape)
    
    return df_overall   

    
### Explore Data
def check_if_features_exist(df, features):
    '''
    Purpose: return False if all features listed
    exist in dataframe df, True otherwise.
    Inputs:
        df (dataframe): data to explore
        features (list): list of features to check

    Returns: (bool): False if all features listed
    exist in dataframe df, True otherwise
    '''

    return any(feature not in df.columns for feature in features)


def get_mean_median_by_group(df, groupby_features, feature):
    '''
    Purpose: produce a dataframe of aggregations

    Inputs:
        df (dataframe): data to explore
        groupby_features (list): list of features to group by
        feature (str): a numeric feature to aggregate by

    Returns: (dataframe) a dataframe constructed from aggregations
    '''

    if check_if_features_exist(df, groupby_features) or not groupby_features:
        return None

    overall_mean = df[feature].mean()
    ndf = df.groupby(groupby_features)[feature]
    ndf = ndf.agg([('median', np.median), ('mean', np.mean),
                   ('diff_from_group_mean', lambda x: x.mean() - overall_mean)])

    return ndf


def get_rates_by_group(df, groupby_features, feature):
    '''
    Purpose: returns dataframe of rates given in outcome column

    Inputs:
        df (dataframe): data to explore
        groupby_features (list): list of features to group by
        feature (str): a numeric feature to get rates

    Returns: (dataframe) dataframe with the rates
    '''
    
    if check_if_features_exist(df, groupby_features + feature) or not groupby_features:
        return None

    by_feature = df.groupby(groupby_features + feature).size()
    by_groupby_features = by_feature / df.groupby(groupby_features).size()

    return by_groupby_features.unstack(fill_value=0)


def check_dataframe_datatypes(df, hide=False, very_short=True, short=False):
    '''
    Purpose: Go through all variables in the DataFrame and print out 
    basic information regarding each column: datatype, counts, missing values.
    
    Inputs:
        df (dataframe): data to explore

    Returns: None, function only prints out the info'''
    
    summary = pd.DataFrame(columns=['column', 'type', 'unique values', 'NA values', 'percent NA', 'examples'])

    print("DataFrame shape: ", df.shape, "\n\n")

    for v in df.columns:
        if df[v].dtype == 'O':
            if pd.DataFrame(df[v].value_counts()).empty:
                ex = None
            else:
                ex = str(pd.DataFrame(df[v].value_counts()).index[0])
            summ = {'column': [v], 'type': ['Object'], 'unique values': [len(df[v].unique())], 
            'NA values': [df[df[v].isna()].shape[0]], 'percent NA': [round(df[df[v].isna()].shape[0]/df.shape[0], 2)],
            'examples': [ex]}
            summary = summary.append(pd.DataFrame.from_dict(summ), ignore_index=True)

            if not hide:
                print(v, 'is Object with ', len(df[v].unique()), ' unique values')
                print("NA counts: ", df[df[v].isna()].shape[0])
                if very_short:
                    print(pd.DataFrame(df[v].value_counts())[:9])
                elif short:
                    print(pd.DataFrame(df[v].value_counts())[:99])
                else:
                    print(pd.DataFrame(df[v].value_counts()))
                print("\n\n")

    for v in df.columns:
        if df[v].dtype == 'bool':
            if pd.DataFrame(df[v].value_counts()).empty:
                ex = None
            else:
                ex = str(pd.DataFrame(df[v].value_counts()).index[0])
            summ = {'column': [v], 'type': ['bool'], 'unique values': [len(df[v].unique())], 
            'NA values': [df[df[v].isna()].shape[0]], 'percent NA': [round(df[df[v].isna()].shape[0]/df.shape[0], 2)],
            'examples': [ex]}
            summary = summary.append(pd.DataFrame.from_dict(summ), ignore_index=True)

            if not hide:
                print(v, 'is Bool')
                print("NA counts: ", df[df[v].isna()].shape[0])
                if very_short:
                    print(pd.DataFrame(df[v].value_counts())[:9])
                elif short:
                    print(pd.DataFrame(df[v].value_counts())[:99])
                else:
                    print(pd.DataFrame(df[v].value_counts()))
                print("\n\n")

    for v in df.columns:
        if df[v].dtype in ('float64', 'int64'):
            if pd.DataFrame(df[v].value_counts()).empty:
                ex = None
            else:
                ex = str(pd.DataFrame(df[v].value_counts()).index[0])
            summ = {'column': [v], 'type': ['Numeric'], 'unique values': [len(df[v].unique())], 
            'NA values': [df[df[v].isna()].shape[0]], 'percent NA': [round(df[df[v].isna()].shape[0]/df.shape[0], 2)],
            'examples': [ex]}
            summary = summary.append(pd.DataFrame.from_dict(summ), ignore_index=True)

            if not hide:
                print(v, 'is Numeric')
                print("NA counts: ", df[df[v].isna()].shape[0])
                if len(pd.DataFrame(df[v].value_counts())) < 10:
                    print(pd.DataFrame(df[v].value_counts()))
                else:
                    print(pd.DataFrame(df[v].describe()))
                print("\n\n")
    
    for v in df.columns:
        if df[v].dtype in ('datetime64', '<M8[ns]'):
            if pd.DataFrame(df[v].value_counts()).empty:
                ex = None
            else:
                ex = str(pd.DataFrame(df[v].value_counts()).index[0])
            summ = {'column': [v], 'type': ['DateTime'], 'unique values': [len(df[v].unique())], 
            'NA values': [df[df[v].isna()].shape[0]], 'percent NA': [round(df[df[v].isna()].shape[0]/df.shape[0], 2)],
            'examples': [ex]}
            summary = summary.append(pd.DataFrame.from_dict(summ), ignore_index=True)

            if not hide:
                print(v, 'is DateTime')
                print("NA counts: ", df[df[v].isna()].shape[0])
                if len(pd.DataFrame(df[v].value_counts())) < 10:
                    print(pd.DataFrame(df[v].value_counts()))
                else:
                    print(pd.DataFrame(df[v].describe()))
                print("\n\n")

    for v in df.columns:
        if df[v].dtype not in ('O', 'bool', 'float64', 'int64', 'datetime64', '<M8[ns]'):
            if not df[v][0]:
                ex = None
            else:
                ex = str(df[v][0])
            summ = {'column': [v], 'type': ['Some Other Data Types'], 'unique values': [len(df[v].unique())], 
            'NA values': [df[df[v].isna()].shape[0]], 'percent NA': [round(df[df[v].isna()].shape[0]/df.shape[0], 2)],
            'examples': [ex]}
            summary = summary.append(pd.DataFrame.from_dict(summ), ignore_index=True)

            if not hide:
                print(v, 'is Some Other Data Types')
                try:
                    print(pd.DataFrame(df[v].describe()))
                    print("NA counts: ", df[df[v].isna()].shape[0])
                    print("\n\n")
                except:
                    print('Data Type cannot be described')

    return summary


def check_timeseries_features(df, var):
    '''
    Purpose: check if the data has hourly, monthly, or yearly data
    Inputs:
        df (dataframe): data to explore
        var: variable in question

    Returns: Strings to describe
    '''

    print('No hourly data:')
    for v in df[var].unique():
        if len(df[df[var] == v]['Hourly'].unique()) == 1:
            print(v, "(",  len(df[df[var] == v]), "observations ) only has Hourly value as:", df[df[var] == v]['Hourly'].unique())
            
    print('No daily data:')
    for v in df[var].unique():
        if len(df[df[var] == v]['Daily'].unique()) == 1:
            print(v, "(",  len(df[df[var] == v]), "observations ) only has Daily value as:", df[df[var] == v]['Daily'].unique())
            
    print('No monthly data:')
    for v in df[var].unique():
        if len(df[df[var] == v]['Monthly'].unique()) == 1:
            print(v, "(",  len(df[df[var] == v]), "observations ) only has Monthly value as:", df[df[var] == v]['Monthly'].unique())

    print('No yearly data:')
    for v in df[var].unique():
        if len(df[df[var] == v]['Yearly'].unique()) == 1:
            print(v, "(",  len(df[df[var] == v]), "observations ) only has Yearly value as:", df[df[var] == v]['Yearly'].unique())


def plot_feature_timeseries(df, time_feature, data_feature,
    to_count=None, to_sum=None, to_avg=None,
    title=None, figsize=None, ax=None, barplot=False):
    '''
    Purpose: plot timeseries of a given feature
    Inputs:
        df (dataframe): data to explore
        time_feature: the time variable in the data
        data_feature: feature to show timeseries for

    Returns: None, the function only plots
    '''

    if figsize:
        fig = plt.figure(figsize=figsize)

    if barplot:
        if to_count:
            temp = df[[time_feature, data_feature]].groupby([time_feature]).count().reset_index()
            g = sns.barplot(x = time_feature, y = data_feature, data = temp, ax = ax, color = '#00BFFF')
            g.set(ylabel =  "Count of " + data_feature, title = title)
        elif to_sum: 
            temp = df[[time_feature, data_feature]].groupby([time_feature]).sum().reset_index()
            g = sns.barplot(x = time_feature, y = data_feature, data = temp, ax = ax, color = '#00BFFF')
            g.set(ylabel = "Sum of " + data_feature, title = title)
        elif to_avg: 
            temp = df[[time_feature, data_feature]].groupby([time_feature]).mean().reset_index()
            g = sns.barplot(x = time_feature, y = data_feature, data = temp, ax = ax, color = '#00BFFF')
            g.set(ylabel = "Average of " + data_feature, title = title)
        else:
            g = sns.barplot(x = time_feature, y = data_feature, data = df, ax = ax, color = '#00BFFF')
            g.set(ylabel = "Raw values of " + data_feature, title = title)
        
        g.figure.autofmt_xdate()

    else:
        if to_count:
            temp = df[[time_feature, data_feature]].groupby([time_feature]).count().reset_index()
            g = sns.lineplot(x = time_feature, y = data_feature, data = temp, ax = ax)
            g.set(ylabel =  "Count of " + data_feature, title = title)
        elif to_sum: 
            temp = df[[time_feature, data_feature]].groupby([time_feature]).sum().reset_index()
            g = sns.lineplot(x = time_feature, y = data_feature, data = temp, ax = ax)
            g.set(ylabel = "Sum of " + data_feature, title = title)
        elif to_avg: 
            temp = df[[time_feature, data_feature]].groupby([time_feature]).mean().reset_index()
            g = sns.lineplot(x = time_feature, y = data_feature, data = temp, ax = ax)
            g.set(ylabel = "Average of " + data_feature, title = title)
        else:
            g = sns.lineplot(x = time_feature, y = data_feature, data = df, ax = ax)
            g.set(ylabel = "Raw values of " + data_feature, title = title)

        g.figure.autofmt_xdate()

    return g


def plot_multiple_timeseries(df, catvar, valvar, plots, n_rows=1,
    to_count=None, to_sum=None, to_avg=None, figsize=(25, 5)):
    
    for v in df[catvar].unique():
        fig, axs = plt.subplots(n_rows, (len(plots) + 1) // n_rows, figsize=figsize)
        ct = 0
        for i in range(len(axs)):
            if isinstance(axs[i], collections.Iterable):
                for j in range(len(axs[i])):
                    if ct < len(plots):
                        plot_feature_timeseries(df[df[catvar] == v],
                                                    plots[ct], valvar,
                                                    to_count=to_count, to_sum=to_sum, to_avg=to_avg,
                                                    title=v, ax=axs[i][j])
                        ct = ct + 1
            else:
                if ct < len(plots):
                    plot_feature_timeseries(df[df[catvar] == v],
                                                plots[ct], valvar,
                                                to_count=to_count, to_sum=to_sum, to_avg=to_avg,
                                                title=v, ax=axs[i])
                    ct = ct + 1

        fig.autofmt_xdate()

    return fig


def plot_features_hist(df, feature, by=None, figsize=None, title=None, ax=None, bins=50):
    '''
    Purpose: Plot the histogram of the feature listed
    Inputs:
        df (dataframe): data to explore
        feature (str): list of features (or list of lists) to plot histograms 
        by (str): categorical variable to plot on the same plot (if needed)

    Returns: None, the function only plots'''

    if figsize:
        fig = plt.figure(figsize=figsize)

    if by:
        for v in df[by].unique():
            g = sns.distplot(df[df[by]==v][feature], label=feature + ' - ' + v,
                bins=bins, ax=ax)
            if title: 
                g.set(title=title)
            else:
                g.set(title='Histogram of ' + features)
    else:
        g = sns.distplot(df[feature], label=feature,
            bins=bins, ax=ax)
        if title: 
            g.set(title=title)
        else:
            g.set(title='Histogram of ' + features)

    return g


def plot_features_pairplot(df, features, hue=None, plot_kws={"alpha": 0.1}):
    '''
    Purpose: plot correlations between features

    Inputs:
        df (dataframe): data to explore
        features (list): list of features to plot correlations
        hue (str): if want to split by some variable

    Returns: None, function only plots
    '''

    if hue:
        features = features + [hue]
        sdf = df[features]
    else:
        sdf = df[features]
        
    sns.pairplot(sdf, hue=hue, plot_kws=plot_kws)

    return None


def summary_timeseries(df, time_feature, cat_feature=None, val_feature=None,
                      to_count=None, to_sum=None, to_avg=None, figsize=(30,15)):
    check_dataframe_datatypes(df)
    convert_to_timeseries(df, [time_feature])
    create_timeseries_features(df, time_feature)
    check_dataframe_datatypes(df)
    
    if cat_feature:
        check_timeseries_features(df, cat_feature)
    
    if val_feature:
        plot_multiple_timeseries(df, cat_feature, val_feature, 
                                      ['Hourly', 'Monthly', 'Yearly', 'MonthYear', 'Daily'] + [time_feature], 2,
                                      to_count=to_count, to_sum=to_sum, to_avg=to_avg, figsize=figsize)
        



### Create Training and Testing Sets
def split_train_test(df, test_size = 0.2, random_state = 1):
    '''
    Purpose: split df into train and test sets

    Inputs:
    df (dataframe): data to split
    test_size, random_state: config of the split

    Returns:
    train, test (dataframe): train, test sets
    '''

    train, test = train_test_split(df)
    print('train set: ', train.shape)
    print('test set: ', test.shape)

    return train, test


### Pre-Process Data for Training
def apply_category_filters(df, filter_info):
    '''
    Purpose: apply filters to categorical features in the df

    Inputs:
        df (dataframe)
        filter_info (dict): of the form {'feature_name':
            ['value1', 'value2', ...]}

    Returns: (dataframe) a new filtered dataframe,
      or None if a specified column does not exist
    '''

    if check_if_features_exist(df, filter_info.keys()):
        return None

    for col in filter_info:
        ndf = df[df[col].isin(filter_info[col])]

    return ndf


def apply_range_filters(df, filter_info):
    '''
    Purpose: apply filters to numeric features in the df

    Inputs:
        df (dataframe)
        filter_info (dict): of the form {'column_name': ['value1', 'value2']}

    Returns: (dataframe) a new filtered dataframe,
      or None if a specified column does not exist
    '''

    if check_if_features_exist(df, filter_info.keys()):
        return None

    for col in filter_info:
        ndf = df[(df[col] >= filter_info[col][0])
                & (df[col] <= filter_info[col][1])]

    return ndf


def process_bool_and_missing(train, test, bool_, bool_na, num, num_na, cat, cat_na):
    '''
    Purpose: apply filters to numeric features in the df

    Inputs:
        df (dataframe)
        filter_info (dict): of the form {'column_name': ['value1', 'value2']}

    Returns: (dataframe) filtered dataframe,
      or None if a specified column does not exist
    '''

    for f in bool_:
        if train[f].dtype == 'bool':
            #print(f, "is bool, converting to int")
            train[f] = train[f].astype(int)
            if test:
                test[f] = test[f].astype(int)
        if bool_na:
            train[f] = train[f].fillna(bool_na)
            if test:
                test[f] = test[f].fillna(bool_na)

    for f in num:
        if train[f].dtype in ('float64', 'int64'):   
            #print(f, "training data's mean:", train[f].mean(),
                #"will replace missing values of", f)
            if not num_na:
                train[f][train[f].isna()] = train[f].mean()
                if test:
                    test[f][test[f].isna()] = train[f].mean()
            else:
                train[f] = train[f].fillna(num_na)
                if test:
                    test[f] = test[f].fillna(num_na)              

    for f in cat:
        if train[f].dtype == 'O':   
            #print(f, "training data's mean:", train[f].mean(),
                #"will replace missing values of", f)
            if cat_na:
                train[f] = train[f].fillna(cat_na)
                if test:
                    test[f] = test[f].fillna(cat_na)      

    if not test:
        return train
    return train, test


def normalize_features(train, test, num):
    '''
    Purpose: normalize the set of features listed, using training set
    mean and standard deviation

    Inputs:
    train, test (df): train and test sets

    Returns: modify the existing train and test sets with new normalized 
    variables as new variables
    '''

    for f in num:
        scaler = StandardScaler()
        scaler.fit(pd.DataFrame(train.loc[:, f]))
        n_f = 'Norm ' + f
        train[n_f] = scaler.transform(pd.DataFrame(train.loc[:, f]))
        if test:
            test[n_f] = scaler.transform(pd.DataFrame(test.loc[:, f]))

    if not test:
        return train
    return train, test


def rescale_features(train, test, num):
    '''
    Purpose: normalize the set of features listed, using training set
    mean and standard deviation

    Inputs:
    train, test (df): train and test sets

    Returns: modify the existing train and test sets with new normalized 
    variables as new variables
    '''

    for f in num:
        n_f = 'Rescaled ' + f
        f_min = train[f].min()
        f_max = train[f].max()
        train[n_f] = (train[f] - f_min) / (f_max - f_min) 
        if test:
            test[n_f] = (test[f] - f_min) / (f_max - f_min) 

    if not test:
        return train
    return train, test


### Generate Features
def one_hot_encoding_features(train, test, cat, prefix):
    '''
    Purpose: Encode categorical variables

    Inputs:
    train, test (df): train and test sets
    features (list): list of features to encode

    Returns: modify the existing train and test sets

    '''

    train = pd.get_dummies(train, columns = cat, prefix = prefix)

    if test:
        test = pd.get_dummies(test, columns = cat, prefix = prefix)
    
        for f in test.columns:
            if f not in train.columns:
                test = test.drop(columns=[f])
            
        for f in train.columns:
            if f not in test.columns:
                test[f] = 0

    if not test:
        return train
    return train, test


class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)
        self.fit_flag = False

    def fit(self, X, **kwargs):
        out = super().fit(X)
        self.fit_flag = True
        return out

    def transform(self, X, **kwargs):
        sparse_matrix = super(OneHotEncoder, self).transform(X)
        new_columns = self.get_new_columns(X=X)
        d_out = pd.DataFrame(sparse_matrix.toarray(), columns=new_columns, index=X.index)
        #return sparse_matrix.toarray(), new_columns 
        return d_out

    def fit_transform(self, X, **kwargs):
        self.fit(X)
        return self.transform(X)

    def get_new_columns(self, X):
        new_columns = []
        for i, column in enumerate(X.columns):
            j = 0
            while j < len(self.categories_[i]):
                new_columns.append(f'{column}_<{self.categories_[i][j]}>')
                j += 1
        return new_columns


def prepare_train_test(train, test, num, cat, bool_, bool_na):

    if test:
        train, test = process_bool_and_missing(train, test, bool_, bool_na, num)
        train, test = normalize_features(train, test, num)
        train, test = one_hot_encoding_features(train, test, cat, cat)

        return train, test
    
    train = process_bool_and_missing(train, test, bool_, bool_na, num)
    train = normalize_features(train, test, num)
    train = one_hot_encoding_features(train, test, cat, cat)

    return train


def temporal_train_test_split(df, train_yr, test_yr, num, cat):
    
    train = df.loc[df.Year.isin(train_yr), :]
    test = df.loc[df.Year.isin(test_yr), :]
    #print('TRAIN: ', train_yr, 'TEST: ', test_yr)
    #print('Training size: ', train.shape) 
    #print('Testing size: ', test.shape) 
    train, test = prepare_train_test(train, test, num, cat)
    
    return train, test


def discretize_features(train, test, feature, bins, quantile=True, labels=False, right=True):
    '''
    Purpose: Discretize continuous variables

    Inputs:
    train, test (df): train and test sets
    features (list): list of features to discretize
    bins_tuples (list): list of tuples to discretize
                for example: [(0, 1), (2, 3), (4, 5)]
                or int, which indicates how many bins to create

    Returns: modify the existing train and test sets

    '''
    
    #if type(bins_tuples) == list:
        #bins = pd.IntervalIndex.from_tuples(bins_tuples)
    

    if quantile:
        n_f = 'Discrete ' + feature
        train[n_f] = pd.qcut(train[feature], bins, labels=labels)

        if test:
            test[n_f] = pd.cut(test[feature], bins, labels=labels)

    if not quantile:
        n_f = 'Discrete ' + feature
        train[n_f] = pd.cut(train[feature], bins, labels=labels, right=right)

        if test:
            test[n_f] = pd.cut(test[feature], bins, labels=labels, right=right)

    return None


### Build Classifiers
#Write 1 function in pipeline.py that applies at least one machine learning 
#model to a dataset. The function should also print the amount of time required 
#to train each model. Several scikit-learn methods will be useful here, including 
#set_params and model_selection.ParameterGrid.

def build_classifiers(train_features, train_targets, model, params):
    '''
    Purpose: Apply machine learning model to training data

    Inputs:
    train_features (df): dataframe of training features data
    train_targets (array): 1d array of training target data
    model: class of model to fit
    params: params of the model

    Returns: model object
    '''
    
    # Begin timer 
    start = datetime.datetime.now()
       
    # Create model 
    print("Training model:", model, "|", params) 
    model.set_params(**params)
            
    # Fit model on training set 
    model.fit(train_features, train_targets)
                
    # End timer
    stop = datetime.datetime.now()
    print("Time Elapsed:", stop - start)  

    return model  


### Evaluate Classifiers
def evaluate_classifiers(features, targets, model):
    '''
    Purpose: Evaluate a built model on some data using 
    sklearn built in mean accuracy score

    Inputs:
    features (df): dataframe of features data
    targets (array): 1d array of target data
    model: model object built from previous steps

    Returns: (float) mean accuracy score
    '''    

    # Predict on features 
    model.predict(features)
            
    # Evaluate predictions 
    score = model.score(features, targets)
            
    return score
