'''
Retrieve the relevant data from the database
to be displayed in a table as results.
'''

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


DATA_DIR = os.path.dirname(__file__)
#DATABASE_FILENAME = os.path.join(DATA_DIR, 'db.sqlite3')
OUTDIR = os.path.join(DATA_DIR, 'data')
INDIR = os.path.join(OUTDIR, 'downloaded')
NEIGH_DIR = os.path.join(OUTDIR, 'neighs')
TEMP_DIR = '/Users/ldinh/Documents/GitHub/Air-Quality-Tool/data/points/'
STATIC = os.path.join(DATA_DIR, 'static/graphs')


def unload_geo(filename):
    df = pd.read_csv(os.path.join(OUTDIR, filename))
    df['geometry'] = df['geometry'].apply(wkt.loads)
    df = gpd.GeoDataFrame(df, geometry='geometry')
    df.crs = "EPSG:4326"
    
    return df


def plot_summary_by_neighborhood(df, timevar):

    by_neigh = df[[timevar] + ['pri_neigh', 'sum', 'count']].groupby([timevar] + ['pri_neigh']).sum().reset_index()
    if timevar:
        by_neigh['timevar_count'] = 1
    else:
        by_neigh['timevar_count'] = by_neigh['count']

    by_neigh = by_neigh.groupby(['pri_neigh']).sum().reset_index()
    by_neigh['mean'] = by_neigh['sum']/by_neigh['count']
    by_neigh = by_neigh.sort_values(by='mean')
    by_neigh = by_neigh.set_index('pri_neigh')
    y1 = df['thirtymins'].dt.year.min()
    y2 = df['thirtymins'].dt.year.max()

    unit_dict = {'Daily': 'days', 'MonthYear': 'months', 'Year': 'years'}

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    width = .35

    by_neigh['mean'].plot(kind='bar', color='red', ax=ax1, width=width, position=0)
    by_neigh['timevar_count'].plot(kind='bar', color='green', ax=ax2, width=width, position=1)

    ax1.set_ylabel('Average PM2.5')
    ax1.set_xlabel('Neighborhood')
    ax1.set_xlim(-1, len(by_neigh))
    if timevar:
        ax2.set_ylabel('Number of ' + unit_dict[timevar] + ' with measurements ' + '(' + str(y1) + '-' + str(y2) + ')')
    else:
        ax2.set_ylabel('Number of measurements ' + '(' + str(y1) + '-' + str(y2) + ')')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if timevar:
        ax2.legend(lines1 + lines2, ['Average PM2.5', 'Number of ' + unit_dict[timevar] + ' with measurements ' + '(' + str(y1) + '-' + str(y2) + ')'], loc='upper center')
    else:
        ax2.legend(lines1 + lines2, ['Average PM2.5', 'Number of measurements ' + '(' + str(y1) + '-' + str(y2) + ')'], loc='upper center')

    fig.savefig('static/graphs/neighborhood_summary_{}.png'.format(timevar), transparent=False, bbox_inches='tight')


def plot_avg_figures(combined, var_list, label_list, scat_x, scat_y, line_ax1, line_ax2, f):
    years = [x for x in list(combined.index.year.unique()) if x > 2017]
    for y in years:
        temp = combined.dropna(0)
        temp = temp[temp.index.month.isin([4,5,6,7,8,9]) & (temp.index.year.isin(years))]
        #temp = temp.resample('D').asfreq()
        vmin = min(temp[var_list].min())
        vmax = max(temp[var_list].max())
        temp = temp[temp.index.month.isin([4,5,6,7,8,9]) & (temp.index.year == y)]
        
        cell_s = len(temp.index.unique())
        figure, axes = plt.subplots(figsize=(4, 0.25*cell_s))
        sns.heatmap(temp[var_list], annot=True, linewidths=.5, ax=axes, cmap="Reds", vmin=vmin, vmax=vmax, fmt='.1f', cbar_kws={"shrink": 0.5})
        
        axes.axes.set_title("Daily Avg PM2.5: Summer " + str(y), fontsize=10, y=1.01)
        axes.axes.set_xlabel("Source", labelpad=15, rotation=0)
        axes.axes.set_xticklabels(labels=label_list, rotation=0)
        axes.axes.set_ylabel("Date", labelpad=20, rotation=0)
        axes.axes.set_yticklabels(labels=temp.index.strftime('%Y-%m-%d').sort_values().unique())
        
        figure.savefig('static/graphs/{}/Heatmap_Daily_Avg_PM25_Summer_{}.png'.format(f, y), transparent=False, bbox_inches='tight')
        #plt.yticks(rotation=0)
 
    for y in years:
        temp = combined.dropna(0)
        temp = temp[temp.index.month.isin([4,5,6,7,8,9]) & (temp.index.year.isin(years))]
        #temp = temp.resample('D').sum().fillna(0)
        temp = temp[temp.index.month.isin([4,5,6,7,8,9]) & (temp.index.year == y)]
        figure = pipeline.plot_2_axis(temp, temp.index, {'ax1': line_ax1, 'ax2': line_ax2},
                'Daily Avg PM2.5', '', title='Summer ' + str(y), figsize=(12,6))
        figure.savefig('static/graphs/{}/Timeseries_Daily_Avg_PM25_Summer_{}.png'.format(f, y), transparent=False, bbox_inches='tight')

    for y in years:
        temp = combined.dropna(0)
        temp = temp[temp.index.month.isin([4,5,6,7,8,9]) & (temp.index.year.isin(years))]
        #temp = temp.resample('D').sum().fillna(0)
        temp = temp[temp.index.month.isin([4,5,6,7,8,9]) & (temp.index.year == y)]
        
        for v1 in scat_y:
            figure = pipeline.plot_scatter(temp, scat_x, v1, hue=None, title='Daily Avg PM2.5: Summer ' + str(y), figsize=(8,8))
            figure.savefig('static/graphs/{}/Scatter_Daily_Avg_PM25_{}_vs_{}_by_{}.png'.format(f, scat_x, v1, y), transparent=False, bbox_inches='tight')

    for v1 in scat_y:
        for v2 in [('Year', (7,7)), (None, (8,8))]:
            temp = combined.dropna(0)
            temp = temp[temp.index.month.isin([4,5,6,7,8,9]) & (temp.index.year.isin(years))]
            figure = pipeline.plot_scatter(temp, scat_x, v1, hue=v2[0], title='Daily Avg PM2.5: All years', figsize=v2[1])
            figure.savefig('static/graphs/{}/Scatter_Daily_Avg_PM25_{}_vs_{}_by_{}.png'.format(f, scat_x, v1, v2[0]), transparent=False, bbox_inches='tight')

    for y in years:
        temp = combined.dropna(0)
        temp = temp[temp.index.month.isin([4,5,6,7,8,9]) & (temp.index.year.isin([2018, 2019, 2020]))]
        temp = temp[temp.index.month.isin([4,5,6,7,8,9]) & (temp.index.year == y)]
        
        temp = pd.melt(temp, id_vars=['Daily'], value_vars=var_list)
        figure, ax = plt.subplots(figsize=(5, 10))

        sns.boxplot(y="value", x="variable", data=temp,
                    whis=[0, 100], palette="vlag", ax=ax)
        sns.swarmplot(y="value", x="variable", data=temp,
                      size=2, color=".3", linewidth=0)

        ax.yaxis.grid(True)
        ax.set(xlabel="Summer " + str(y))
        ax.set_xticklabels(labels=label_list)
        ax.set(ylabel="Daily Avg PM2.5")
        sns.despine(trim=True, left=True)
        figure.savefig('static/graphs/{}/Boxplot_Daily_Avg_PM25_Summer_{}.png'.format(f, y), transparent=False, bbox_inches='tight')
    
    temp = combined.dropna(0)
    temp = temp[temp.index.month.isin([4,5,6,7,8,9]) & (temp.index.year.isin(years))]
    temp = pd.melt(temp, id_vars=['Daily'], value_vars=var_list)
    fig, ax = plt.subplots(figsize=(5, 10))

    sns.boxplot(y="value", x="variable", data=temp,
                    whis=[0, 100], palette="vlag", ax=ax)
    sns.swarmplot(y="value", x="variable", data=temp,
                      size=2, color=".3", linewidth=0)

    ax.yaxis.grid(True)
    ax.set(xlabel="All years")
    ax.set_xticklabels(labels=label_list)
    ax.set(ylabel="Daily Avg PM2.5")
    sns.despine(trim=True, left=True)
    fig.savefig('static/graphs/{}/Boxplot_Daily_Avg_PM25_All_Summers.png'.format(f), transparent=False, bbox_inches='tight')
    

def plot_harmful_figures(combined, harmful, harmful_lab, not_harmful, not_harmful_lab, f):  
    years = [x for x in list(combined.index.year.unique()) if x > 2017]
    for y in years:
        temp = combined.dropna(0)
        temp = temp[temp.index.month.isin([4,5,6,7,8,9]) & (temp.index.year.isin(years))]
        #temp = temp.resample('D').asfreq()
        vmin = min(temp[harmful].min())
        vmax = max(temp[harmful].max())
        temp = temp[temp.index.month.isin([4,5,6,7,8,9]) & (temp.index.year == y)]

        cell_s = len(temp.index.unique())
        figure, axes = plt.subplots(figsize=(4, 0.25*cell_s))
        sns.heatmap(temp[harmful], annot=True, linewidths=.5, ax=axes, cmap="Reds", vmin=vmin, vmax=0.1, fmt='.1%', cbar_kws={"shrink": 0.5})

        axes.axes.set_title("PM2.5 Measurements >=35 (%): Summer " + str(y), fontsize=10, y=1.01)
        axes.axes.set_xlabel("Source", labelpad=15, rotation=0)
        axes.axes.set_xticklabels(labels=harmful_lab, rotation=0)
        axes.axes.set_ylabel("Date", labelpad=20, rotation=0)
        axes.axes.set_yticklabels(labels=temp.index.strftime('%Y-%m-%d').sort_values().unique())
        
        figure.savefig('static/graphs/{}/Heatmap_Daily_Harmful_PM25_Summer_{}.png'.format(f, y), transparent=False, bbox_inches='tight')
        #plt.yticks(rotation=0)
    
    for y in years:
        temp = combined.dropna(0)
        temp = temp[temp.index.month.isin([4,5,6,7,8,9]) & (temp.index.year.isin(years))]
        #temp = temp.resample('D').asfreq()
        vmin = min(temp[not_harmful].min())
        vmax = max(temp[not_harmful].max())
        temp = temp[temp.index.month.isin([4,5,6,7,8,9]) & (temp.index.year == y)]
        
        cell_s = len(temp.index.unique())
        figure, axes = plt.subplots(figsize=(4, 0.25*cell_s))
        sns.heatmap(temp[not_harmful], annot=True, linewidths=.5, ax=axes, cmap="Greens", vmin=vmin, vmax=vmax, fmt='.1%', cbar_kws={"shrink": 0.5})

        axes.axes.set_title("PM2.5 Measurements <=12 (%): Summer " + str(y), fontsize=10, y=1.01)
        axes.axes.set_xlabel("Source", labelpad=15, rotation=0)
        axes.axes.set_xticklabels(labels=not_harmful_lab, rotation=0)
        axes.axes.set_ylabel("Date", labelpad=20, rotation=0)
        axes.axes.set_yticklabels(labels=temp.index.strftime('%Y-%m-%d').sort_values().unique())
        
        figure.savefig('static/graphs/{}/Heatmap_Daily_Low_PM25_Summer_{}.png'.format(f, y), transparent=False, bbox_inches='tight')
        #plt.yticks(rotation=0)
        
    #for y in [2018, 2019, 2020]:
        #temp = combined.dropna(0)
        #temp = temp.resample('D').sum().fillna(0)
        #temp = temp[temp.index.month.isin([4,5,6,7,8,9]) & (temp.index.year == y)]
        #figure = process_data.plot_2_axis(temp, temp.index, {'ax1': [('AQ_harmful', 'green', '-'), ('EPA_harmful', 'coral', '--'), ('PA_harmful', 'yellow', '--')], 'ax2': None},
                #'PM2.5 Measurements >=35 (%)', '', title='Summer ' + str(y), figsize=(16,8))
     
        #figure.savefig('static/graphs/{}/Timeseries_Daily_Harmful_PM25_Summer_{}.png'.format(f, y))
        
    #for y in [2018, 2019, 2020]:
        #temp = combined.dropna(0)
        #temp = temp.resample('D').sum().fillna(0)
        #temp = temp[temp.index.month.isin([4,5,6,7,8,9]) & (temp.index.year == y)]
        #figure = process_data.plot_2_axis(temp, temp.index, {'ax1': [('AQ_not_harmful', 'green', '-'), ('EPA_not_harmful', 'coral', '--'), ('PA_not_harmful', 'yellow', '--')], 'ax2': None},
                #'PM2.5 Measurements <=12 (%)', '', title='Summer ' + str(y), figsize=(16,8))
        
        #figure.savefig('static/graphs/{}/Timeseries_Daily_Low_PM25_Summer_{}.png'.format(f, y))


def plot_daily_maps(epa, purpleair, aq_hex, days):

    fig = px.choropleth_mapbox(aq_hex, geojson=aq_hex, color="Average PM2.5", opacity=0.8,
                               locations="id", featureidkey="properties.id",
                               animation_frame='Date',
                               range_color=[0,30],
                               color_continuous_scale='YlOrRd', zoom=13,
                               category_orders={'Yearly': ['All', '2016', '2017', '2018', '2019', '2020'], 
                                             'Hourly': ['All', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
                                             '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24'], 
                                             'Monthly': ['All', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
                                             'MonthYear': ['All'],
                                             'Date': days,
                                             'Average PM2.5 Group': ["0-6", "6-12", "12-22", "22-35", "35+"]},
                               hover_name='Source',
                               hover_data={'Average PM2.5': True, 'id': False, 'Date': False},
                               center={"lat": 41.8781, "lon": -87.6298})

    fig2 = px.choropleth_mapbox(epa, geojson=epa, color='Average PM2.5', opacity=0.5,
                        locations="site_number", featureidkey="properties.site_number",
                        #size='count',
                        #hover_name="value_represented", 
                        hover_name='Source',
                        hover_data={'Average PM2.5': True, 'Date': False, 'site_number': False, 'EPA Station': True},
                        range_color=[0,30],
                        color_continuous_scale='YlOrRd', zoom=13,
                        animation_frame='Date')
    fig2.update_traces(marker_line_width=1)

    fig3 = px.scatter_mapbox(purpleair, lat='lat', lon='lon', color='Average PM2.5', opacity=0.8,
                        #size=12,
                        hover_name='Source',
                        hover_data={'Average PM2.5': True, 'Date': False, 'lat': False, 'lon': False},
                        range_color=[0,30],
                        color_continuous_scale='YlOrRd', zoom=13,
                        animation_frame='Date')
    fig3.update_traces(marker=dict(size=15, symbol='circle'))

    fig.add_trace(fig2.data[0], secondary_y=False)
    fig.add_trace(fig3.data[0], secondary_y=False)
    for i, frame in enumerate(fig.frames):
        fig.frames[i].data += (fig2.frames[i].data[0],)
        fig.frames[i].data += (fig3.frames[i].data[0],)
    fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, legend=dict(yanchor="bottom", xanchor="left"), width=1200, height=600)

    return fig


def plot_neigh(aq_points_neigh_agg, timevar):
    lat_cent = round(aq_points_neigh_agg['lat_r'].median(), 4)
    lon_cent = round(aq_points_neigh_agg['lng_r'].median(), 4)

    fig = px.scatter_mapbox(aq_points_neigh_agg, 
                        lat='lat_r', lon='lng_r',
                        color='Average PM2.5 Group', #size='count',
                        #color_discrete_sequence=['yellow'],
                        #color_continuous_scale=px.colors.cyclical.IceFire, 
                        color_discrete_map = {'0-6': "#31a354",
                                              '6-12': "#bae4b3",
                                              '12-22': "#fc8d59",
                                              '22-35': "#e34a33",
                                              '35+': "#b30000"},
                        zoom=13,
                        animation_frame=timevar,
                        #hover_name='Average PM2.5 Group',
                        hover_data={'mean': True, 'time': True, 'Average PM2.5 Group': False, 'lat_r': False, 'lng_r': False, timevar: False},
                        category_orders={'Yearly': ['All', '2016', '2017', '2018', '2019', '2020'], 
                                         'Hourly': ['All', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
                                         '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24'], 
                                         'Monthly': ['All', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
                                         'MonthYear': ['All'],
                                         'Average PM2.5 Group': ["0-6", "6-12", "12-22", "22-35", "35+"]},
                        center={"lat": lat_cent, "lon": lon_cent})

    fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, legend=dict(yanchor="bottom", xanchor="left"), width=600, height=600)

    return fig


def plot_neigh_block(aq_points_neigh_agg, timevar):
    lat_cent = round(aq_points_neigh_agg['geometry'].centroid.y.median(), 4)
    lon_cent = round(aq_points_neigh_agg['geometry'].centroid.x.median(), 4)      

    fig = px.choropleth_mapbox(aq_points_neigh_agg, geojson=aq_points_neigh_agg, 
                            locations='geoid10', featureidkey="properties.geoid10",
                            color='Average PM2.5 Group', #size='count',
                            #color_discrete_sequence=['yellow'],
                            #color_continuous_scale=px.colors.cyclical.IceFire, 
                            color_discrete_map = {'Low 90%': "#ffffcc",
                                                  'Top 10%': "#b30000"},
                            zoom=13,
                            animation_frame=timevar,
                            #hover_name='Average PM2.5 Group',
                            hover_data={'mean': True, 'count': True, 'geoid10': False, 'Average PM2.5 Group': False, timevar: False},
                            category_orders={'Yearly': ['All', '2016', '2017', '2018', '2019', '2020'], 
                                             'Hourly': ['All', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
                                             '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24'], 
                                             'Monthly': ['All', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
                                             'MonthYear': ['All'],
                                             'Average PM2.5 Group': ['Highest 10%', 'Remaining 90%']},
                            center={"lat": lat_cent, "lon": lon_cent})

    fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, legend=dict(yanchor="bottom", xanchor="left"), width=600, height=600)

    return fig

