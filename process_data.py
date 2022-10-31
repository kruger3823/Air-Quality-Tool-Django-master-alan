# Process data specific functions
# Author: Linh Dinh

import os
import csv
import json
import requests
import scipy

import importlib
import pipeline
import itertools
import datetime
from datetime import timedelta

from scipy import stats
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from matplotlib import colors
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import seaborn as sns
palette = itertools.cycle(sns.color_palette())

from geopandas import GeoDataFrame
import geopandas as gpd

import json
from mpl_toolkits.axes_grid1 import make_axes_locatable

from shapely.geometry import Point
from shapely import wkt

import plotly.express as px
import plotly.graph_objects as go


def create_level_flags(df, harm, not_harm):
    df['harmful'] = np.nan
    df['harmful'][df[harm] >= 35] = 1
    df['harmful'][df[harm] < 35] = 0

    df['not_harmful'] = np.nan
    df['not_harmful'][df[not_harm] <= 12] = 1
    df['not_harmful'][df[not_harm] > 12] = 0    


def create_pm_level_cat(df, harm, no_harm):
    df['harmful_cat'] = ''
    df['harmful_cat'][(df[harm] >= 0) & (df[harm] < 0.01)] = '0-1'
    df['harmful_cat'][(df[harm] > 0.01) & (df[harm] < 0.03)] = '1-3'
    df['harmful_cat'][(df[harm] >= 0.03) & (df[harm] < 0.06)] = '3-6'
    df['harmful_cat'][(df[harm] >= 0.06) & (df[harm] < 0.14)] = '6-14'
    df['harmful_cat'][(df[harm] >= 0.14)] = '14+'

    df['low_cat'] = ''
    df['low_cat'][(df[no_harm] >= 0) & (df[no_harm] < 0.02)] = '0-2'
    df['low_cat'][(df[no_harm] > 0.02) & (df[no_harm] < 0.1)] = '2-10'
    df['low_cat'][(df[no_harm] >= 0.1) & (df[no_harm] < 0.9)] = '10-90'
    df['low_cat'][(df[no_harm] >= 0.9) & (df[no_harm] < 0.98)] = '90-98'
    df['low_cat'][(df[no_harm] >= 0.98)] = '98+'


def process_epa(epa):
    pipeline.convert_to_str(epa, ['state_code', 'county_code', 'site_number', 'parameter_code',
                                  'poc', 'units_of_measure_code', 'sample_duration_code', 'detection_limit',
                                  'method_code', 'cbsa_code', 'code'])
    pipeline.convert_to_flt(epa, ['shape_area', 'shape_len'])
    epa['Full Date'] = epa['date_local'] + ' ' + epa['time_local']
    #pipeline.convert_to_timeseries(epa, ['Full Date'])
    #pipeline.create_timeseries_features(epa, 'Full Date')

    epa_dates = epa[['site_number', 'value_represented', 'latitude', 'longitude', 'sample_duration', 'sample_frequency', 'method_type', 'Full Date']].groupby(
        ['site_number', 'value_represented', 'latitude', 'longitude', 'sample_duration', 'sample_frequency', 'method_type']).agg(
        [('min', np.min), ('max', np.max)]).reset_index()
    epa_sites = epa[['site_number', 'value_represented', 'latitude', 'longitude', 'sample_duration', 'sample_frequency', 'method_type', 'sample_measurement']].groupby(
        ['site_number', 'value_represented', 'latitude', 'longitude', 'sample_duration', 'sample_frequency', 'method_type']).agg(
        [('median', np.median), ('mean', np.mean), ('max', np.max), ('sum', np.sum), ('count', lambda x: x.shape[0])]).reset_index()
    epa_sites = epa_sites.merge(epa_dates, on=['site_number', 'value_represented', 'latitude', 'longitude', 'sample_duration', 'sample_frequency', 'method_type'], how='inner')

    epa_sites_temp = epa_sites[['value_represented', 'longitude', 'latitude']].drop_duplicates()
    epa_sites_temp.columns = epa_sites_temp.columns.droplevel(1)
    epa_sites_gdf = gpd.GeoDataFrame(epa_sites_temp,
                                     geometry=gpd.points_from_xy(epa_sites_temp.longitude, epa_sites_temp.latitude))
    epa_sites_gdf.crs = "EPSG:4326"
    epa_sites_gdf.to_crs(4326)

    buff = epa_sites_gdf.buffer(1/60)
    envelope = buff.envelope 

    epa_sites_gdf['buffer'] = buff
    epa_sites_gdf['envelope'] = envelope
    epa_sites_gdf = epa_sites_gdf[['value_represented', 'envelope']]

    epa_sites.columns = epa_sites.columns.map('|'.join).str.strip('|')
    #epa_sites.columns = epa_sites.columns.to_flat_index()
    epa_sites = epa_sites.merge(epa_sites_gdf, on='value_represented', how='left')
    
    epa = epa[['site_number', 'parameter_code', 'latitude', 'longitude', 'sample_measurement',
           'sample_duration', 'sample_frequency', 'method_type', 'value_represented',
           'Full Date']]

    #epa = epa.merge(epa_sites_gdf, how='left', on='value_represented')
    #create_level_flags(epa, 'sample_measurement', 'sample_measurement')
    return epa, epa_sites


def process_purpleair(purpleair):

    purpleair[['lat', 'lon']] = purpleair['location'].astype(str).str.split(pat=',', n=1, expand=True)
    purpleair['lat'] = purpleair['lat'].str.replace('(', '').astype(float)
    purpleair['lon'] = purpleair['lon'].str.replace(')', '').astype(float)
    purpleair['ParentID'][purpleair['ParentID'].isnull()] = purpleair['ID']
    purpleair['ParentID'] = purpleair['ParentID'].astype(int)
    pipeline.convert_to_str(purpleair, ['ID', 'ParentID', 'primaryID'])
    #pipeline.convert_to_timeseries(purpleair, ['created_at'])
    #pipeline.create_timeseries_features(purpleair, 'created_at')
    purpleair['device_loc'] = purpleair['device_loc'].fillna('missing')
    purpleair['PM2.5 (ATM)'] = purpleair['PM2.5 (ATM)'].astype(float)

    purpleair_dates = purpleair[['address', 'lat', 'lon', 'name', 'device_loc', 'ID', 'primaryID', 'ParentID', 'sensor_created_at', 'created_at']].groupby(
        ['address', 'lat', 'lon', 'name', 'device_loc', 'ID', 'primaryID', 'ParentID', 'sensor_created_at']).agg(
        [('min', np.min), ('max', np.max)]).reset_index()

    purpleair_sites = purpleair[['address', 'lat', 'lon', 'name', 'device_loc', 'ID', 'primaryID', 'ParentID', 'sensor_created_at', 'PM2.5 (ATM)']].groupby(
        ['address', 'lat', 'lon', 'name', 'device_loc', 'ID', 'primaryID', 'ParentID', 'sensor_created_at']).agg(
        [('median', np.median), ('mean', np.mean), ('max', np.max), ('sum', np.sum), ('count', lambda x: x.shape[0])]).reset_index()

    purpleair_sites = purpleair_sites.merge(purpleair_dates, on=['address', 'lat', 'lon', 'name', 'device_loc', 'ID', 'primaryID', 'ParentID', 'sensor_created_at'], how='inner')

    purpleair_std_add = purpleair_sites[purpleair_sites['ParentID'] == purpleair_sites['ID']][['address', 'ParentID']].drop_duplicates().rename(columns={'address': 'std_address'})
    purpleair_sites = purpleair_sites.merge(purpleair_std_add, on='ParentID', how='left')

    purpleair_sites_temp = purpleair_sites[['address', 'lon', 'lat']].drop_duplicates()
    purpleair_sites_temp.columns=purpleair_sites_temp.columns.droplevel(1)
    purpleair_sites_gdf = gpd.GeoDataFrame(purpleair_sites_temp,
                          geometry=gpd.points_from_xy(purpleair_sites_temp.lon, purpleair_sites_temp.lat))
    purpleair_sites_gdf.crs = "EPSG:4326"
    purpleair_sites_gdf.to_crs(4326)

    buff = purpleair_sites_gdf.buffer(1/60)
    envelope = buff.envelope  

    purpleair_sites_gdf['buffer'] = buff
    purpleair_sites_gdf['envelope'] = envelope
    purpleair_sites_gdf = purpleair_sites_gdf[['address', 'envelope']]

    purpleair_sites.columns = purpleair_sites.columns.map('|'.join).str.strip('|')
    #purpleair_sites.columns = purpleair_sites.columns.to_flat_index()
    purpleair_sites = purpleair_sites.merge(purpleair_sites_gdf, on='address', how='left')
 
    purpleair = purpleair[['address', 'lat', 'lon', 'ID', 'ParentID',
                       'PM2.5 (ATM)', 'PM2.5 (CF=1)', 'created_at']]
    #purpleair = purpleair.merge(purpleair_sites_gdf, how='left', on='address')
    #create_level_flags(purpleair, 'PM2.5 (ATM)', 'PM2.5 (ATM)')

    return purpleair, purpleair_sites


def process_aq(aq_points, blocks, neighborhoods, hexagons, big_hexagons):
    aq_points = gpd.GeoDataFrame(aq_points, geometry=gpd.points_from_xy(aq_points.lng, aq_points.lat))
    aq_points.crs = "EPSG:4326"
    aq_points.to_crs(4326)

    aq_points = gpd.sjoin(aq_points, blocks[['geoid10', 'geometry']], how="left", op='intersects')
    aq_points = aq_points.drop(columns=['index_right'])

    aq_points = gpd.sjoin(aq_points, neighborhoods, how="left", op='intersects')
    aq_points = aq_points.drop(columns=['index_right'])

    aq_points = gpd.sjoin(aq_points, hexagons, how="left", op='intersects')
    aq_points = aq_points.drop(columns=['index_right'])
    aq_points = aq_points.drop(columns=['id_left'])
    aq_points = aq_points.rename(columns={'id_right': 'hexagon_id'})

    aq_points = gpd.sjoin(aq_points, big_hexagons, how="left", op='intersects')
    aq_points = aq_points.rename(columns={'index_right': 'big_hexagon_id'})

    aq_points['time'] = aq_points['time'].str.replace('T', ' ', regex=False)
    pipeline.convert_to_timeseries(aq_points, ['time'])
    aq_points['thirtymins'] = aq_points['time'].dt.floor('30min')
    pipeline.create_timeseries_features(aq_points, 'time')
    create_level_flags(aq_points, 'value', 'value')

    aq_by_block = aq_points[['geoid10', 'thirtymins', 'harmful', 'not_harmful', 'value']].groupby(
    ['geoid10', 'thirtymins', 'harmful', 'not_harmful']).agg(
    [('median', np.median), ('mean', np.mean), ('max', np.max), ('min', np.min), ('sum', np.sum), ('count', lambda x: x.shape[0])]).reset_index()
    aq_by_block.columns = aq_by_block.columns.map('|'.join).str.strip('|').str.replace("value|", "").str.strip('|')

    aq_by_neighborhood = aq_points[['pri_neigh', 'thirtymins', 'harmful', 'not_harmful', 'value']].groupby(
    ['pri_neigh', 'thirtymins', 'harmful', 'not_harmful']).agg(
    [('median', np.median), ('mean', np.mean), ('max', np.max), ('min', np.min), ('sum', np.sum), ('count', lambda x: x.shape[0])]).reset_index()
    aq_by_neighborhood.columns = aq_by_neighborhood.columns.map('|'.join).str.strip('|').str.replace("value|", "").str.strip('|')

    aq_by_hexagon = aq_points[['hexagon_id', 'thirtymins', 'harmful', 'not_harmful', 'value']].groupby(
    ['hexagon_id', 'thirtymins', 'harmful', 'not_harmful']).agg(
    [('median', np.median), ('mean', np.mean), ('max', np.max), ('min', np.min), ('sum', np.sum), ('count', lambda x: x.shape[0])]).reset_index()
    aq_by_hexagon.columns = aq_by_hexagon.columns.map('|'.join).str.strip('|').str.replace("value|", "").str.strip('|')

    aq_by_big_hexagon = aq_points[['big_hexagon_id', 'thirtymins', 'harmful', 'not_harmful', 'value']].groupby(
    ['big_hexagon_id', 'thirtymins', 'harmful', 'not_harmful']).agg(
    [('median', np.median), ('mean', np.mean), ('max', np.max), ('min', np.min), ('sum', np.sum), ('count', lambda x: x.shape[0])]).reset_index()
    aq_by_big_hexagon.columns = aq_by_big_hexagon.columns.map('|'.join).str.strip('|').str.replace("value|", "").str.strip('|')

    return aq_by_block, aq_by_neighborhood, aq_by_hexagon, aq_by_big_hexagon


def create_agg_data(df, timevar, catvar, sumvar, maxvar, valvar, harm=None, not_harm=None, hexa=None):
    df_agg = pipeline.create_summary_timeseries_dataset(df, timevar, catvar, sumvar, maxvar, valvar)

    if harm and not_harm:
        df_agg_h = pipeline.create_summary_timeseries_dataset(df, timevar, catvar + harm + not_harm, sumvar, maxvar, valvar)
        df_agg_h = df_agg_h[timevar + catvar + harm + not_harm + ['count']]

        c = df_agg_h[timevar + catvar + ['count']].groupby(timevar + catvar).sum().reset_index()
        c = c[timevar + catvar + ['count']]

        c1 = df_agg_h[timevar + catvar + harm + ['count']].groupby(timevar + catvar + harm).sum().reset_index()
        c1 = c1[c1[harm[0]] == 1][timevar + catvar + ['count']].rename(columns={'count': harm[0]})

        c2 = df_agg_h[timevar + catvar + not_harm + ['count']].groupby(timevar + catvar + not_harm).sum().reset_index()
        c2 = c2[c2[not_harm[0]] == 1][timevar + catvar + ['count']].rename(columns={'count': not_harm[0]})

        c = c.merge(c1, on=timevar + catvar, how='left')
        c = c.merge(c2, on=timevar + catvar, how='left')

        c[harm[0]] = c[harm[0]].fillna(0)/c['count']
        c[not_harm[0]] = c[not_harm[0]].fillna(0)/c['count']
        c = c[timevar + catvar + harm + not_harm]

        df_agg = df_agg.merge(c, on=timevar + catvar, how='left')

        if hexa:
            df_agg_hexa = pipeline.create_summary_timeseries_dataset(df, timevar, catvar + hexa, sumvar, maxvar, valvar)
            create_level_flags(df_agg_hexa, 'mean', 'mean')

            df_agg_hexa = pipeline.create_summary_timeseries_dataset(df_agg_hexa, timevar, catvar + harm + not_harm, [], [], ['mean'])
            df_agg_hexa = df_agg_hexa[timevar + catvar + harm + not_harm + ['count']]

            c = df_agg_hexa[timevar + catvar + ['count']].groupby(timevar + catvar).sum().reset_index()
            c = c[timevar + catvar + ['count']]

            c1 = df_agg_hexa[timevar + catvar + harm + ['count']].groupby(timevar + catvar + harm).sum().reset_index()
            c1 = c1[c1[harm[0]] == 1][timevar + catvar + ['count']].rename(columns={'count': harm[0]})

            c2 = df_agg_hexa[timevar + catvar + not_harm + ['count']].groupby(timevar + catvar + not_harm).sum().reset_index()
            c2 = c2[c2[not_harm[0]] == 1][timevar + catvar + ['count']].rename(columns={'count': not_harm[0]})

            c = c.merge(c1, on=timevar + catvar, how='left')
            c = c.merge(c2, on=timevar + catvar, how='left')

            c[harm[0]] = c[harm[0]].fillna(0)/c['count']
            c[not_harm[0]] = c[not_harm[0]].fillna(0)/c['count']
            c = c[timevar + catvar + harm + not_harm]
            c = c.rename(columns={harm[0]: harm[0] + '_hex', not_harm[0]: not_harm[0] + '_hex'})

            df_agg = df_agg.merge(c, on=timevar + catvar, how='left')            

    if timevar:
        pipeline.convert_to_timeseries(df_agg, timevar)

    return df_agg


def combine_data_sources(aq, epa, purpleair, timevar):
    aq[timevar] = aq[timevar].astype(str)
    epa[timevar] = epa[timevar].astype(str)
    purpleair[timevar] = purpleair[timevar].astype(str)
    
    df = aq.merge(epa, on=timevar, how='left')
    df = df.rename(columns={'sum_x': 'AQ_sum', 'count_x': 'AQ_count', 'mean_x': 'AQ_mean', 'max_x': 'AQ_max', 'median': 'EPA_median', 
                            'harmful_x': 'AQ_harmful', 'not_harmful_x': 'AQ_not_harmful',
                            'mean_y': 'EPA_mean', 'max_y': 'EPA_max', 'sum_y': 'EPA_sum', 'count_y': 'EPA_count',
                            'harmful_y': 'EPA_harmful', 'not_harmful_y': 'EPA_not_harmful'})

    df = df.merge(purpleair, on=timevar, how='left')
    df = df.rename(columns={'median': 'PA_median', 'mean': 'PA_mean', 'harmful': 'PA_harmful', 'not_harmful': 'PA_not_harmful',
                            'max': 'PA_max', 'sum': 'PA_sum', 'count': 'PA_count'})

    pipeline.convert_to_timeseries(df, [timevar])
    df.set_index(df[timevar], inplace=True)

    return df

