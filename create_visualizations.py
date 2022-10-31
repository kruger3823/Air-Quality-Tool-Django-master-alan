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
import plot_functions

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


def unload_aq(filename):
    df = pd.read_csv(os.path.join(OUTDIR, filename))
    pipeline.convert_to_timeseries(df, ['thirtymins'])
    pipeline.create_timeseries_features(df, 'thirtymins')
    
    return df


def process_daily_data(aq_by_hexagon):

    epa_daily = pd.read_csv(os.path.join(OUTDIR, 'epa_daily.csv'))
    epa_daily = process_data.create_agg_data(epa_daily, ['Daily'], [], ['sum', 'count'], ['max'], [], ['harmful'], ['not_harmful'], [])
    purpleair_outside_only_no_outliers_daily = pd.read_csv(os.path.join(OUTDIR, 'purpleair_outside_daily.csv'))
    purpleair_outside_only_no_outliers_daily = process_data.create_agg_data(purpleair_outside_only_no_outliers_daily, ['Daily'], [], ['sum', 'count'], ['max'], [], ['harmful'], ['not_harmful'], [])

    aq_by_hex_daily = process_data.create_agg_data(aq_by_hexagon, ['Daily'], [], ['sum', 'count'], ['max'], [], ['harmful'], ['not_harmful'], ['hexagon_id'])
    temp = process_data.create_agg_data(aq_by_hexagon, ['Daily'], ['hexagon_id'], ['sum', 'count'], ['max'], [], ['harmful'], ['not_harmful'])
    temp = temp[['Daily', 'hexagon_id']].groupby('Daily').count().reset_index().rename(columns={'hexagon_id': 'hex_ct'})
    aq_by_hex_daily = aq_by_hex_daily.merge(temp, on='Daily', how='left')

    return epa_daily, purpleair_outside_only_no_outliers_daily, aq_by_hex_daily


def process_2_miles_radius_data(hexagons, aq_by_hexagon):

    epa_sites = pd.read_csv(os.path.join(OUTDIR, 'epa_sites.csv'))
    epa_sites['envelope'] = epa_sites['envelope'].apply(wkt.loads)
    epa_sites = gpd.GeoDataFrame(epa_sites, geometry='envelope')

    purpleair_sites = pd.read_csv(os.path.join(OUTDIR, 'purpleair_sites.csv'))
    purpleair_sites['envelope'] = purpleair_sites['envelope'].apply(wkt.loads)
    purpleair_sites = gpd.GeoDataFrame(purpleair_sites, geometry='envelope')

    aq_hex_geo = hexagons[['id', 'geometry']].drop_duplicates()
    aq_hex_geo = gpd.GeoDataFrame(aq_hex_geo, geometry='geometry')
    epa_geo = epa_sites[['value_represented', 'envelope']].drop_duplicates()
    epa_geo = gpd.GeoDataFrame(epa_geo, geometry='envelope')
    purpleair_geo = purpleair_sites[['address', 'envelope']].drop_duplicates()
    purpleair_geo = gpd.GeoDataFrame(purpleair_geo, geometry='envelope')

    epa_aq_hex_xwalk = gpd.sjoin(epa_geo, aq_hex_geo, how="left", op='intersects')
    epa_aq_hex_xwalk = epa_aq_hex_xwalk.drop(columns=['index_right', 'envelope'])
    purpleair_aq_hex_xwalk = gpd.sjoin(purpleair_geo, aq_hex_geo, how="left", op='intersects')
    purpleair_aq_hex_xwalk = purpleair_aq_hex_xwalk.drop(columns=['index_right', 'envelope'])

    aq_within_2_miles_epa = pd.DataFrame(epa_aq_hex_xwalk['id'].unique())
    aq_within_2_miles_epa.columns = ['hexagon_id']
    aq_within_2_miles_epa = aq_by_hexagon.merge(aq_within_2_miles_epa, on='hexagon_id', how='inner')
    aq_within_2_miles_epa_daily = process_data.create_agg_data(aq_within_2_miles_epa, ['Daily'], [], ['sum', 'count'], ['max'], [], ['harmful'], ['not_harmful'], ['hexagon_id'])
    temp = process_data.create_agg_data(aq_within_2_miles_epa, ['Daily'], ['hexagon_id'], ['sum', 'count'], ['max'], [], ['harmful'], ['not_harmful'])
    temp = temp[['Daily', 'hexagon_id']].groupby('Daily').count().reset_index().rename(columns={'hexagon_id': 'hex_ct'})
    aq_within_2_miles_epa_daily = aq_within_2_miles_epa_daily.merge(temp, on='Daily', how='left')
    
    aq_within_2_miles_purpleair = pd.DataFrame(purpleair_aq_hex_xwalk['id'].unique())
    aq_within_2_miles_purpleair.columns = ['hexagon_id']
    aq_within_2_miles_purpleair = aq_by_hexagon.merge(aq_within_2_miles_purpleair, on='hexagon_id', how='inner')
    aq_within_2_miles_purpleair_daily = process_data.create_agg_data(aq_within_2_miles_purpleair, ['Daily'], [], ['sum', 'count'], ['max'], [], ['harmful'], ['not_harmful'], ['hexagon_id'])
    temp = process_data.create_agg_data(aq_within_2_miles_purpleair, ['Daily'], ['hexagon_id'], ['sum', 'count'], ['max'], [], ['harmful'], ['not_harmful'])
    temp = temp[['Daily', 'hexagon_id']].groupby('Daily').count().reset_index().rename(columns={'hexagon_id': 'hex_ct'})
    aq_within_2_miles_purpleair_daily = aq_within_2_miles_purpleair_daily.merge(temp, on='Daily', how='left')

    return aq_within_2_miles_epa_daily, aq_within_2_miles_purpleair_daily 


def create_comparison_data(aq, aq_name, epa, epa_name, pa, pa_name):

    for aq_d in [(aq[(aq['count'] >= 500) & (aq['hex_ct'] >= 5)], aq_name)]:
        for epa_d in [(epa[epa['count'] >= 12], epa_name)]:
            for purpleair_d in [(pa[pa['count'] >= 250], pa_name)]:
                print(aq_d[1], epa_d[1], purpleair_d[1])
                combined = process_data.combine_data_sources(aq_d[0], epa_d[0], purpleair_d[0], 'Daily')
                combined['Year'] = combined['Daily'].dt.year

    return combined


def create_comparison_data_for_maps(aq_by_hexagon, common):

    epa = pd.read_csv(os.path.join(OUTDIR, 'epa_daily.csv'))
    epa_sites = pd.read_csv(os.path.join(OUTDIR, 'epa_sites.csv'))
    epa_sites['envelope'] = epa_sites['envelope'].apply(wkt.loads)
    epa_sites = gpd.GeoDataFrame(epa_sites, geometry='envelope')
    epa_sites.crs = "EPSG:4326"
    epa = epa.merge(epa_sites[['value_represented', 'site_number', 'envelope']].drop_duplicates(), on='value_represented', how='left')
    #epa = epa[['Daily', 'value_represented', 'latitude', 'longitude', 'mean', 'count']]
    #epa = gpd.GeoDataFrame(epa, geometry=gpd.points_from_xy(epa.longitude, epa.latitude))
    epa = epa[['Daily', 'value_represented', 'site_number', 'envelope', 'mean', 'count']]
    epa = gpd.GeoDataFrame(epa, geometry='envelope')
    epa['mean'] = round(epa['mean'], 2)
    epa = epa[epa['mean'].notna()]
    pipeline.convert_to_timeseries(epa, ['Daily'])
    epa['Source'] = 'EPA'
    epa = epa.rename(columns={'value_represented': 'EPA Station', 'mean': 'Average PM2.5', 'Daily': 'Date'})

    purpleair = pd.read_csv(os.path.join(OUTDIR, 'purpleair_outside_daily.csv'))
    purpleair_sites = pd.read_csv(os.path.join(OUTDIR, 'purpleair_sites.csv'))
    purpleair = process_data.create_agg_data(purpleair, ['Daily'], ['lat', 'lon'], ['sum', 'count'], [], [], [], [])
    purpleair = purpleair[['Daily', 'lat', 'lon', 'mean', 'count']]
    purpleair = gpd.GeoDataFrame(purpleair, geometry=gpd.points_from_xy(purpleair.lon, purpleair.lat))
    purpleair['mean'] = round(purpleair['mean'], 2)
    pipeline.convert_to_timeseries(purpleair, ['Daily'])
    purpleair['Source'] = 'PurpleAir'
    purpleair = purpleair.rename(columns={'mean': 'Average PM2.5', 'Daily': 'Date'})

    aq_hex = process_data.create_agg_data(aq_by_hexagon, ['Daily'], ['hexagon_id'], ['sum', 'count'], [], [], [], [])
    aq_hex = aq_hex.merge(hexagons.reset_index(), left_on='hexagon_id', right_on='id', how='inner', suffixes=('_left', '_right'))
    aq_hex = aq_hex[['Daily', 'mean', 'count', 'geometry', 'id']]
    aq_hex = gpd.GeoDataFrame(aq_hex, geometry='geometry')
    aq_hex['mean'] = round(aq_hex['mean'], 2)
    aq_hex['Source'] = 'AirQuality'
    aq_hex = aq_hex.rename(columns={'mean': 'Average PM2.5', 'Daily': 'Date'})

    epa = epa.merge(common[['Daily']].rename(columns={'Daily': 'Date'}), on='Date', how='inner')
    purpleair = purpleair.merge(common[['Daily']].rename(columns={'Daily': 'Date'}), on='Date', how='inner')
    aq_hex = aq_hex.merge(common[['Daily']].rename(columns={'Daily': 'Date'}), on='Date', how='inner')

    return epa, purpleair, aq_hex


def create_neighborhood_lon_lat_data(df):
    
    aq_points_neigh = process_data.create_agg_data(df, ['thirtymins'], ['lat_r', 'lng_r'], ['sum', 'count'], [], [], [], [])
    pipeline.create_timeseries_features(aq_points_neigh, 'thirtymins')
    pipeline.discretize_features(aq_points_neigh, [], 'mean', bins=[-0.1, 6, 12, 22, 35, 1000],
            quantile=False, labels=["0-6", "6-12", "12-22", "22-35", "35+"])
    aq_points_neigh['mean'] = round(aq_points_neigh['mean'], 1)
    for v in ['thirtymins', 'Hourly', 'Daily', 'Monthly', 'MonthYear', 'Yearly']:
        aq_points_neigh[v] = aq_points_neigh[v].astype(str)

    aq_points_neigh_all = aq_points_neigh.drop(columns=['Hourly', 'Daily', 'Monthly', 'MonthYear', 'Yearly'])
    for v in ['Hourly', 'Daily', 'Monthly', 'MonthYear', 'Yearly']:
        aq_points_neigh_all[v] = 'All'
    aq_points_neigh = aq_points_neigh.append(aq_points_neigh_all)

    aq_points_neigh = aq_points_neigh.rename(columns={'Discrete mean': 'Average PM2.5 Group', 'thirtymins': 'time'})  
    aq_points_neigh = aq_points_neigh[['lat_r', 'lng_r', 'Average PM2.5 Group', 'Hourly', 'Daily', 'Monthly', 'MonthYear', 'Yearly', 'time', 'mean']]
    aq_points_neigh = aq_points_neigh.drop_duplicates()

    return aq_points_neigh


def create_neighborhood_block_data(df, timevar):

    aq_points_neigh = process_data.create_agg_data(df, ['thirtymins'], ['geoid10'], ['sum', 'count'], [], [], [], [])
    pipeline.create_timeseries_features(aq_points_neigh, 'thirtymins')
    aq_points_neigh_agg = process_data.create_agg_data(aq_points_neigh, [], [timevar, 'geoid10'], ['sum', 'count'], [], [], [], [])
    quantile = pd.DataFrame(aq_points_neigh_agg.groupby(timevar)['mean'].quantile(0.9)).reset_index().rename(columns={'mean': 'top10'})
    aq_points_neigh_agg = aq_points_neigh_agg.merge(quantile, on=timevar, how='left')

    top10 = aq_points_neigh_agg[aq_points_neigh_agg['mean'] >= aq_points_neigh_agg['top10']]
    top10 = top10.merge(blocks, how='left', on='geoid10')
    top10 = top10[[timevar, 'count', 'mean', 'geoid10', 'geometry']]
    top10['cat'] = 'Top 10%'

    low90 = aq_points_neigh_agg[aq_points_neigh_agg['mean'] < aq_points_neigh_agg['top10']]
    low90 = low90.merge(blocks, how='left', on='geoid10')
    low90 = low90[[timevar, 'count', 'mean', 'geoid10', 'geometry']]
    low90['cat'] = 'Low 90%'

    aq_points_neigh_agg = top10.append(low90)
    aq_points_neigh_agg['mean'] = round(aq_points_neigh_agg['mean'], 2)

    aq_points_neigh_agg[timevar] = aq_points_neigh_agg[timevar].astype(str)
    aq_points_neigh_agg = gpd.GeoDataFrame(aq_points_neigh_agg, geometry='geometry').reset_index()

    aq_points_neigh_agg = aq_points_neigh_agg[['geoid10', 'geometry',
                                'cat', timevar, 'mean', 'count']]
    aq_points_neigh_agg = aq_points_neigh_agg.rename(columns={'cat': 'Average PM2.5 Group'})       

    return aq_points_neigh_agg


def create_neighborhood_timeseries_data(df, timevar):

    aq_points_neigh = process_data.create_agg_data(df, ['thirtymins'], [], ['sum', 'count'], [], [], [], [])
    pipeline.create_timeseries_features(aq_points_neigh, 'thirtymins')
    aq_points_neigh_agg = process_data.create_agg_data(aq_points_neigh, [], [timevar], ['sum', 'count'], [], [], [], [])
    aq_points_neigh_agg = aq_points_neigh_agg.rename(columns={'mean': 'PM2.5'})
    
    return aq_points_neigh_agg


if __name__ == "__main__":

    print('Load GEO data\n')
    blocks = unload_geo("blocks.csv")
    neighborhoods = unload_geo("neighborhoods.csv")
    hexagons = unload_geo("hexagons.csv")
    big_hexagons = unload_geo("big_hexagons.csv")

    print('Load AQ data\n')
    aq_by_block = unload_aq('aq_by_block.csv')
    aq_by_neighborhood = unload_aq('aq_by_neighborhood.csv')
    aq_by_hexagon = unload_aq('aq_by_hexagon.csv')
    aq_by_big_hexagon = unload_aq('aq_by_big_hexagon.csv')

    print('Process daily data\n')
    epa_daily, purpleair_outside_only_no_outliers_daily, aq_by_hex_daily = process_daily_data(aq_by_hexagon)
    print('Process 2 miles data\n')
    aq_within_2_miles_epa_daily, aq_within_2_miles_purpleair_daily = process_2_miles_radius_data(hexagons, aq_by_hexagon)
    
    print('Plot aq_within_2_miles_epa_daily\n')
    combined1 = create_comparison_data(aq_within_2_miles_epa_daily, 'aq_within_2_miles_epa_daily',
                                epa_daily, 'epa_daily',
                                purpleair_outside_only_no_outliers_daily, 'purpleair_outside_only_no_outliers_daily')
    plot_functions.plot_avg_figures(combined1, ['EPA_mean', 'AQ_mean'], ['EPA', 'AirQuality'],
                             'AQ_mean', ['EPA_mean'],
                             [('AQ_mean', 'green', '-'), ('EPA_mean', 'coral', '--')], None, 'aq_within_2_miles_epa')
    plot_functions.plot_harmful_figures(combined1, ['EPA_harmful', 'AQ_harmful'], ['EPA', 'AirQuality'],
                                ['EPA_not_harmful', 'AQ_not_harmful'], ['EPA', 'AirQuality'], 'aq_within_2_miles_epa')

    print('Plot aq_within_2_miles_purpleair_daily\n')
    combined2 = create_comparison_data(aq_within_2_miles_purpleair_daily, 'aq_within_2_miles_purpleair_daily',
                                epa_daily, 'epa_daily',
                                purpleair_outside_only_no_outliers_daily, 'purpleair_outside_only_no_outliers_daily')
    plot_functions.plot_avg_figures(combined2, ['PA_mean', 'AQ_mean'], ['PurpleAir', 'AirQuality'],
                             'AQ_mean', ['PA_mean'],
                             [('AQ_mean', 'green', '-'), ('PA_mean', 'yellow', '--')], None, 'aq_within_2_miles_pa')
    plot_functions.plot_harmful_figures(combined2, ['PA_harmful', 'AQ_harmful'], ['PurpleAir', 'AirQuality'],
                                ['PA_not_harmful', 'AQ_not_harmful'], ['PurpleAir', 'AirQuality'], 'aq_within_2_miles_pa')
    
    print('Plot aq_by_hex_daily\n')
    combined3 = create_comparison_data(aq_by_hex_daily, 'aq_by_hex_daily',
                                epa_daily, 'epa_daily',
                                purpleair_outside_only_no_outliers_daily, 'purpleair_outside_only_no_outliers_daily')
    plot_functions.plot_avg_figures(combined3, ['EPA_mean', 'AQ_mean', 'PA_mean'], ['EPA', 'AirQuality', 'PurpleAir'],
                             'AQ_mean', ['EPA_mean', 'PA_mean'],
                             [('AQ_mean', 'green', '-'), ('EPA_mean', 'coral', '--'), ('PA_mean', 'yellow', '--')], None, 'aq')
    plot_functions.plot_harmful_figures(combined3, ['EPA_harmful', 'AQ_harmful', 'PA_harmful'], ['EPA', 'AirQuality', 'PurpleAir'],
                                ['EPA_not_harmful', 'AQ_not_harmful', 'PA_not_harmful'], ['EPA', 'AirQuality', 'PurpleAir'], 'aq')

    print('Plot Daily Maps')
    epa_map, purpleair_map, aq_hex_map = create_comparison_data_for_maps(aq_by_hexagon, combined3)

    for y in (2017, 2018, 2019, 2020):
        for m in (4,5,6,7,8,9):
            print('Month, Year: ', m, y)
            days = []
            s = datetime.date(y, 1, 1)
            for i in range(0, 370):
                days.append(s.strftime("%Y-%m-%d"))
                s += timedelta(days=1)

            aq_hex_tmp = aq_hex_map[(aq_hex_map['Date'].dt.year == y) & (aq_hex_map['Date'].dt.month.isin([m]))]
            epa_tmp = epa_map[(epa_map['Date'].dt.year == y) & (epa_map['Date'].dt.month.isin([m]))]
            purpleair_tmp = purpleair_map[(purpleair_map['Date'].dt.year == y) & (purpleair_map['Date'].dt.month.isin([m]))]

            if (len(aq_hex_tmp) > 0) and (len(epa_tmp) > 0) and (len(purpleair_tmp) > 0):
                aq_hex_tmp['Date'] = aq_hex_tmp['Date'].dt.strftime("%Y-%m-%d")
                epa_tmp['Date'] = epa_tmp['Date'].dt.strftime("%Y-%m-%d")
                purpleair_tmp['Date'] = purpleair_tmp['Date'].dt.strftime("%Y-%m-%d")   

                daily_map = plot_functions.plot_daily_maps(epa_tmp, purpleair_tmp, aq_hex_tmp, days)
                daily_map.write_html(os.path.join(STATIC, 'aq', 'comparison_daily_maps_month_' + str(m) + '_year_' + str(y) + '.html'), include_plotlyjs=False)

    print('Plot summary by neighborhood (daily)\n')
    plot_functions.plot_summary_by_neighborhood(aq_by_neighborhood, 'Daily')
    
    print('Plot neighborhood\n')
    neighs = list(neighborhoods['pri_neigh'].unique())
    for n in neighs:
        if os.path.isfile(os.path.join(NEIGH_DIR, n + '.csv')):
            neigh_points = pd.read_csv(os.path.join(NEIGH_DIR, n + '.csv'))
            
            neigh_data_lon_lat = create_neighborhood_lon_lat_data(neigh_points)
            print('Plot neighborhood lon_lat_monthly: ', n, '\n')
            monthly = plot_functions.plot_neigh(neigh_data_lon_lat, 'Monthly')
            monthly.write_html(os.path.join(STATIC, 'neighs', n + '_lon_lat_monthly.html'), auto_play=False, include_plotlyjs=False)
            print('Plot neighborhood lon_lat_hourly: ', n, '\n')
            hourly = plot_functions.plot_neigh(neigh_data_lon_lat, 'Hourly')
            hourly.write_html(os.path.join(STATIC, 'neighs', n + '_lon_lat_hourly.html'), auto_play=False, include_plotlyjs=False)
            
            print('Plot neighborhood block_yearly: ', n, '\n')
            neigh_data_block_yearly = create_neighborhood_block_data(neigh_points, 'Yearly')
            yearly = plot_functions.plot_neigh_block(neigh_data_block_yearly, 'Yearly')
            yearly.write_html(os.path.join(STATIC, 'neighs', n + '_block_yearly.html'), auto_play=False, include_plotlyjs=False)
            print('Plot neighborhood block_monthyear: ', n, '\n')
            neigh_data_block_monthyear = create_neighborhood_block_data(neigh_points, 'MonthYear')
            monthyear = plot_functions.plot_neigh_block(neigh_data_block_monthyear, 'MonthYear')
            monthyear.write_html(os.path.join(STATIC, 'neighs', n + '_block_monthyear.html'), auto_play=False, include_plotlyjs=False)
            
            print('Plot neighborhood timeseries_yearly: ', n, '\n')
            neigh_data_timeseries_yearly = create_neighborhood_timeseries_data(neigh_points, 'Yearly')
            fig = pipeline.plot_feature_timeseries(neigh_data_timeseries_yearly, 'Yearly', 'PM2.5', to_avg=True, figsize=(15, 10), barplot=True)
            fig.figure.savefig(os.path.join(STATIC, 'neighs', n + '_timeseries_yearly.png'), transparent=False, bbox_inches='tight')
            print('Plot neighborhood timeseries_monthyear: ', n, '\n')
            neigh_data_timeseries_monthyear = create_neighborhood_timeseries_data(neigh_points, 'MonthYear')
            fig = pipeline.plot_feature_timeseries(neigh_data_timeseries_monthyear, 'MonthYear', 'PM2.5', to_avg=True, figsize=(15, 10), barplot=True)
            fig.figure.savefig(os.path.join(STATIC, 'neighs', n + '_timeseries_monthyear.png'), transparent=False, bbox_inches='tight')
            
            