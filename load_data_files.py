'''
Crawl EPA, PurpleAir, AirQuality and retrieve all relevant data.
Then, combine this data with appropriate geo/shape files to create master data files.
Store these master data files as .csv files in './data'.
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
OUTDIR = os.path.join(DATA_DIR, 'data')
INDIR = os.path.join(OUTDIR, 'downloaded')
NEIGH_DIR = os.path.join(OUTDIR, 'neighs')
TEMP_DIR = '/Users/ldinh/Documents/GitHub/Air-Quality-Tool/data/points/'


def load_epa(start_date, end_date):

    start_date = start_date.replace('-', '')
    end_date = end_date.replace('-', '')

    SITE_PATH = 'https://aqs.epa.gov/data/api/list/sitesByCounty?email=ldinh@uchicago.edu&key=greyram77&state=17&county=031'
    site_response = requests.get(SITE_PATH)
    site_dict = json.loads(site_response.text)
    site_data = site_dict['Data']

    epa_sites = pd.DataFrame(site_data)
    epa_sites['code'] = epa_sites['code'].astype(int)

    PM_PATH = 'https://aqs.epa.gov/data/api/sampleData/byCounty?email=ldinh@uchicago.edu&key=greyram77&param=88101,88502&bdate=' + start_date + '&edate=' + end_date + '&state=17&county=031'
    pm_response = requests.get(PM_PATH)
    pm_dict = json.loads(pm_response.text)
    pm_data = pm_dict['Data']

    epa = pd.DataFrame(pm_data)
    epa['site_number'] = epa['site_number'].astype(int) 

    epa = epa.merge(epa_sites, left_on='site_number', right_on='code', how='left', suffixes=('_left', '_right'))
    epa, epa_sites = process_data.process_epa(epa)

    return epa, epa_sites


def agg_epa(epa, epa_sites):

    #epa_sites['envelope'] = epa_sites['envelope'].apply(wkt.loads)
    epa_sites = gpd.GeoDataFrame(epa_sites, geometry='envelope')
    epa_sites_env = epa_sites[['value_represented', 'envelope']]
    epa = epa.merge(epa_sites_env, on='value_represented', how='left')

    pipeline.convert_to_timeseries(epa, ['Full Date'])
    pipeline.create_timeseries_features(epa, 'Full Date')
    process_data.create_level_flags(epa, 'sample_measurement', 'sample_measurement')
    
    epa_daily = process_data.create_agg_data(epa, ['Daily'], ['value_represented', 'latitude', 'longitude'], [], [], ['sample_measurement'], ['harmful'], ['not_harmful'])

    return epa_daily, epa_sites


def load_purpleair(start_date, end_date):

    print("Getting PurpleAir sites")
    PA_SITE = 'https://www.purpleair.com/json'
    pa_site_response = requests.get(PA_SITE)
    pa_site_dict = json.loads(pa_site_response.text)   

    print("Getting Chicago GEO limit")
    boundary = gpd.read_file("https://data.cityofchicago.org/resource/y6yq-dbs2.geojson?$limit=9999999")[['geometry']]
    boundary['temp'] = 1
    boundary = boundary.dissolve(by='temp')
    boundary = boundary[['geometry']]
    boundary.crs = "EPSG:4326"

    xmin, ymin, xmax, ymax = boundary.total_bounds
    chicago_sites = []

    for i in pa_site_dict['results']:
        if i.get('Lon') and i.get('Lat'):
            if (xmin <= i['Lon'] <= xmax) & (ymin <= i['Lat'] <= ymax):
                chicago_sites.append(i)

    primary = []
    secondary = []
    pri_cols = set(['ID', 'ParentID', 'address', 'location', 'device_loc', 'name', 'primaryID',
                    'sensor_created_at', 'last_updated_at', 'created_at'])
    sec_cols = set(['ID', 'ParentID', 'address', 'location', 'device_loc', 'name', 'primaryID',
                    'sensor_created_at', 'last_updated_at', 'created_at'])

    print("Looping through PurpleAir sites")
    ct = 0
    for s in chicago_sites:
        print(s['Label'], end = "\n")
        start_date = pd.to_datetime(start_date, format='%Y-%m-%d').date()
        end_date = pd.to_datetime(end_date, format='%Y-%m-%d').date()
        date = start_date

        pri_data = []
        sec_data = []
        
        for i in range((end_date - start_date).days):
            if (ct % 3000 == 0) & ct !=0:
                print("Sleeping")
                time.sleep(400)

            PRI_PATH = 'https://api.thingspeak.com/channels/' + s['THINGSPEAK_PRIMARY_ID'] + '/feed.json?api_key=' + s['THINGSPEAK_PRIMARY_ID_READ_KEY'] + '&offset=0&average=30&round=2&start=' + str(date) + '%20' + '00:00:00&end=' + str(date) + '%20' +'23:59:59'
            ct = ct + 1
            pri_response = requests.get(PRI_PATH)
            pri_dict = json.loads(pri_response.text) 
            pri_data = pri_data + pri_dict['feeds']

            SEC_PATH = 'https://api.thingspeak.com/channels/' + s['THINGSPEAK_SECONDARY_ID'] + '/feed.json?api_key=' + s['THINGSPEAK_SECONDARY_ID_READ_KEY'] + '&offset=0&average=30&round=2&start=' + str(date) + '%20' + '00:00:00&end=' + str(date) + '%20' +'23:59:59'
            ct = ct + 1
            sec_response = requests.get(SEC_PATH)
            sec_dict = json.loads(sec_response.text) 
            sec_data = sec_data + sec_dict['feeds']

            if i == 0:
                pri_header = pri_dict['channel']
                sec_header = sec_dict['channel']

            date += timedelta(days=1)
     
        r = {'ID': s.get('ID'), 'ParentID': s.get('ParentID'), 'address': s.get('Label'),
             'location': (s.get('Lat'), s.get('Lon')), 'device_loc': s.get('DEVICE_LOCATIONTYPE')}
        
        # PRIMARY
        print("Start Primary channel")
        rp = {'name': pri_header.get('name'), 'primaryID': pri_header.get('id'),
              'sensor_created_at': pri_header.get('created_at'), 'last_updated_at': pri_header.get('updated_at')}
            
        for pi in pri_data:
            pi_dict = {}
            pi_dict['created_at'] = pi.get('created_at')  
            for ph in pri_header.keys():
                if re.findall(r'field\w+', ph):
                    pi_dict[pri_header[ph]] = pi.get(ph)   
                    pri_cols.add(pri_header[ph])
        
            pi_dict.update(rp) 
            pi_dict.update(r)
            primary.append(pi_dict)
        
        # SECONDARY
        print("Start Secondary channel")
        rs = {'name': sec_header.get('name'), 'primaryID': sec_header.get('id'),
              'sensor_created_at': sec_header.get('created_at'), 'last_updated_at': sec_header.get('updated_at')}
            
        for si in sec_data:
            si_dict = {}
            si_dict['created_at'] = si.get('created_at')  
            for sh in sec_header.keys():
                if re.findall(r'field\w+', sh):
                    si_dict[sec_header[sh]] = si.get(sh)   
                    sec_cols.add(sec_header[sh])

            si_dict.update(rs) 
            si_dict.update(r)
            secondary.append(si_dict) 

    primary = pd.DataFrame(primary)
    secondary = pd.DataFrame(secondary)

    purpleair, purpleair_sites = process_data.process_purpleair(primary)
    
    outside_only_add = list(purpleair_sites[purpleair_sites['device_loc'] == 'outside']['address'])
    outside_only_id = list(purpleair_sites[purpleair_sites['device_loc'] == 'outside']['ID'])
    purpleair_outside_only = purpleair[purpleair['address'].isin(outside_only_add) | purpleair['ParentID'].isin(outside_only_id)]
    purpleair_outside_only_no_outliers = pipeline.remove_outliers(purpleair_outside_only, ['PM2.5 (ATM)'])

    return purpleair_outside_only_no_outliers, purpleair_sites


def agg_purpleair(purpleair, purpleair_sites):

    #purpleair_sites['envelope'] = purpleair_sites['envelope'].apply(wkt.loads)
    purpleair_sites = gpd.GeoDataFrame(purpleair_sites, geometry='envelope')
    purpleair_sites_env = purpleair_sites[['address', 'envelope']]
    purpleair = purpleair.merge(purpleair_sites_env, on='address', how='left')

    pipeline.convert_to_timeseries(purpleair, ['created_at'])
    pipeline.create_timeseries_features(purpleair, 'created_at')
    process_data.create_level_flags(purpleair, 'PM2.5 (ATM)', 'PM2.5 (ATM)')
    
    purpleair_outside_only_no_outliers_daily = process_data.create_agg_data(purpleair, ['Daily'], ['address', 'lat', 'lon'], [], [], ['PM2.5 (ATM)'], ['harmful'], ['not_harmful'])
    purpleair_outside_only_no_outliers_daily = purpleair_outside_only_no_outliers_daily[purpleair_outside_only_no_outliers_daily['Daily'].dt.date >= datetime.date(2017, 10, 1)]

    return purpleair_outside_only_no_outliers_daily, purpleair_sites


def load_aq(start_date, end_date, download=True):

    if download:
        prefs = {"download.default_directory": TEMP_DIR, "download.directory_upgrade": True}
        options = webdriver.ChromeOptions()
        options.add_experimental_option("prefs", prefs)
        driver = webdriver.Chrome('./chromedriver', chrome_options = options)
        driver.get("https://airqualitychicago.org/data_values/")

        geo_options = driver.find_element_by_id("id_geo_type").find_elements_by_tag_name("option")

        for option in geo_options:
            if option.get_attribute("text") == 'Neighborhood': 
                option.click()

        try:
            geo = WebDriverWait(driver, 60).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="id_geo_boundaries"]/option[1]')))
            neighs = driver.find_element_by_id("id_geo_boundaries").find_elements_by_tag_name("option")
            neighs_list = []
            for n in neighs:
                neighs_list.append(n.get_attribute("text"))
            print("total neighborhoods:", len(neighs_list))
            
        except:
            print("Can't find neighborhood options")

        for neigh in neighs_list:
            print(neigh, start_date, end_date)
            driver.get("https://airqualitychicago.org/data_values/")
            driver.find_element_by_id("id_all_users").click()

            all_options = driver.find_elements_by_tag_name("option")
            for option in all_options:
                if option.get_attribute("text") in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'Neighborhood']: 
                    option.click()

            start_e = driver.find_elements_by_id("id_start_date")[1]
            end_e = driver.find_elements_by_id("id_end_date")[1]
            #driver.execute_script("arguments[0].removeAttribute('readonly')", start_e)
            #driver.execute_script("arguments[0].removeAttribute('readonly')", end_e)
            #ActionChains(driver).move_to_element(start_e).click().send_keys('2017-01-01').perform() 
            #ActionChains(driver).move_to_element(end_e).click().send_keys('2020-07-31').perform() 
            
            try:
                geo = WebDriverWait(driver, 60).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="id_geo_boundaries"]/option[1]')))
                neighs = driver.find_element_by_id("id_geo_boundaries").find_elements_by_tag_name("option")
                
                for n in neighs:
                    if n.get_attribute("text") == neigh:
                        n.click()
                        print(n.get_attribute("text"), start_date, end_date)

                        driver.execute_script("arguments[0].setAttribute('value', '" + start_date + "')", start_e) 
                        driver.execute_script("arguments[0].setAttribute('value', '" + end_date + "')", end_e) 

                        driver.find_element_by_id("id_submit").click()
                        time.sleep(10)
                        
                        try:
                            element = WebDriverWait(driver, 300).until(EC.element_to_be_clickable((By.ID, 'id_download')))
                            element.click()
                            time.sleep(10)
                            print("Done")
                            
                        except:
                            print("Can't load map")
                            
            except:
                print("Can't find neighborhood options")
                    
        driver.quit()
        aq_points = agg_aq()
    
    else: 
        aq_points = agg_aq()

    return aq_points


def agg_aq():

    aq_points = pd.DataFrame()
    ct = 0
    for file in os.listdir(TEMP_DIR):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            temp = pd.read_csv(os.path.join(TEMP_DIR, filename))
            aq_points = aq_points.append(temp)
            ct = ct + 1
            
    return aq_points


def unload_geo(filename):
    df = pd.read_csv(os.path.join(OUTDIR, filename))
    df['geometry'] = df['geometry'].apply(wkt.loads)
    df = gpd.GeoDataFrame(df, geometry='geometry')
    df.crs = "EPSG:4326"
    
    return df


def truncate_dates(df, timevar, cutoff):
    new_df = df[df[timevar] <= pd.to_datetime(cutoff, format='%Y-%m-%d')]

    return new_df


def neigh_filter(df, neigh):

    temp = df[df['pri_neigh'] == neigh]
    #temp = temp[['time', 'value', 'lat', 'lng']]
    temp = temp.reset_index()
    temp = temp.rename(columns = {'index': 'id'})

    return temp


def create_neighborhood_data(df, neigh):

    aq_points_neigh = neigh_filter(df, neigh)
    aq_points_neigh['time'] = aq_points_neigh['time'].str.replace('T', ' ', regex=False)
    pipeline.convert_to_timeseries(aq_points_neigh, ['time'])
    aq_points_neigh['thirtymins'] = aq_points_neigh['time'].dt.floor('30min')
    pipeline.create_timeseries_features(aq_points_neigh, 'time')
    process_data.create_level_flags(aq_points_neigh, 'value', 'value')    

    aq_points_neigh['lat_r'] = round(aq_points_neigh['lat'], 5)
    aq_points_neigh['lng_r'] = round(aq_points_neigh['lng'], 5)

    aq_points_neigh_agg = process_data.create_agg_data(aq_points_neigh, ['thirtymins'], ['geoid10', 'lat_r', 'lng_r'], [], [], ['value'], [], [])

    return aq_points_neigh_agg


def get_new_dates():

    current_epa = pd.read_csv(os.path.join(OUTDIR, 'epa_daily.csv'))
    pipeline.convert_to_timeseries(current_epa, ['Daily'])

    current_purpleair = pd.read_csv(os.path.join(OUTDIR, 'purpleair_outside_daily.csv'))
    pipeline.convert_to_timeseries(current_purpleair, ['Daily'])

    current_aq_by_block = pd.read_csv(os.path.join(OUTDIR, 'aq_by_block.csv'))
    pipeline.convert_to_timeseries(current_aq_by_block, ['thirtymins'])

    current_aq_by_neighborhood = pd.read_csv(os.path.join(OUTDIR, 'aq_by_neighborhood.csv'))
    pipeline.convert_to_timeseries(current_aq_by_neighborhood, ['thirtymins'])

    current_aq_by_hexagon = pd.read_csv(os.path.join(OUTDIR, 'aq_by_hexagon.csv'))
    pipeline.convert_to_timeseries(current_aq_by_hexagon, ['thirtymins'])

    current_aq_by_big_hexagon = pd.read_csv(os.path.join(OUTDIR, 'aq_by_big_hexagon.csv'))
    pipeline.convert_to_timeseries(current_aq_by_big_hexagon, ['thirtymins'])

    neigh_last_updated = datetime.datetime(2016,1,1,0)
    for file in os.listdir(NEIGH_DIR):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            temp = pd.read_csv(os.path.join(NEIGH_DIR, filename))
            pipeline.convert_to_timeseries(temp, ['thirtymins'])
            neigh_date = temp['thirtymins'].max()
            if neigh_date > neigh_last_updated:
                neigh_last_updated = neigh_date

    last_updated = min(current_epa['Daily'].max(),
                      current_purpleair['Daily'].max(),
                      current_aq_by_block['thirtymins'].max(),
                      current_aq_by_neighborhood['thirtymins'].max(),
                      current_aq_by_hexagon['thirtymins'].max(),
                      current_aq_by_big_hexagon['thirtymins'].max(),
                      neigh_last_updated).date()

    last_week = datetime.datetime.now().date() - timedelta(days=14)

    current_epa = truncate_dates(current_epa, 'Daily', last_updated)
    current_purpleair = truncate_dates(current_purpleair, 'Daily', last_updated)
    current_aq_by_block = truncate_dates(current_aq_by_block, 'thirtymins', last_updated)
    current_aq_by_neighborhood = truncate_dates(current_aq_by_neighborhood, 'thirtymins', last_updated)
    current_aq_by_hexagon = truncate_dates(current_aq_by_hexagon, 'thirtymins', last_updated)
    current_aq_by_big_hexagon = truncate_dates(current_aq_by_big_hexagon, 'thirtymins', last_updated)

    return last_updated, last_week, current_epa, current_purpleair, current_aq_by_block, current_aq_by_neighborhood, current_aq_by_hexagon, current_aq_by_big_hexagon


if __name__ == "__main__":
    # total arguments 
    
    last_updated, last_week, current_epa, current_purpleair, current_aq_by_block, current_aq_by_neighborhood, current_aq_by_hexagon, current_aq_by_big_hexagon = get_new_dates()
    
    if len(sys.argv) == 3:
        n = len(sys.argv) 
        print("Total arguments passed:", n) 
        # Arguments passed 
        print("\nName of Python script:", sys.argv[0]) 
        print("\nArguments passed:", end = "\n")
        for i in range(1, n): 
            print(sys.argv[i], end = " ") 
        start_date = sys.argv[1]
        end_date = sys.argv[2]
    else: 
        start_date = str(last_updated)
        end_date = str(last_week)
        
    print('start date ', start_date, ' end date ', end_date)
    
    print('Archive old data')
    for name in ['epa_daily.csv', 'epa_sites.csv', 'purpleair_outside_daily.csv', 'purpleair_sites.csv',
                'aq_by_big_hexagon.csv', 'aq_by_hexagon.csv', 'aq_by_block.csv', 'aq_by_neighborhood.csv']:
        os.rename(os.path.join(OUTDIR, name), os.path.join(OUTDIR, 'old', name))
    
    print("Loading EPA data\n")
    epa, epa_sites = load_epa(start_date, end_date)
    print("Aggregating EPA data\n")
    epa_daily, epa_sites = agg_epa(epa, epa_sites)
    print("Saving EPA data\n")
    epa_sites.to_csv(os.path.join(OUTDIR, 'epa_sites.csv'), index=False)
    current_epa = current_epa.append(epa_daily)
    current_epa.to_csv(os.path.join(OUTDIR, 'epa_daily.csv'), index=False)
    
    print("Loading PurpleAir data\n")
    purpleair_outside_only_no_outliers, purpleair_sites = load_purpleair(start_date, end_date)
    print("Aggregating PurpleAir data\n")
    purpleair_outside_only_no_outliers_daily, purpleair_sites = agg_purpleair(purpleair_outside_only_no_outliers, purpleair_sites)
    print("Saving PurpleAir data\n")
    purpleair_sites.to_csv(os.path.join(OUTDIR, 'purpleair_sites.csv'), index=False)
    current_purpleair = current_purpleair.append(purpleair_outside_only_no_outliers_daily)
    current_purpleair.to_csv(os.path.join(OUTDIR, 'purpleair_outside_daily.csv'), index=False)
    
    print("Loading GEO data\n")
    blocks = unload_geo("blocks.csv")
    neighborhoods = unload_geo("neighborhoods.csv")
    hexagons = unload_geo("hexagons.csv")
    big_hexagons = unload_geo("big_hexagons.csv")
    print("Loading Airbeam data\n")
    aq_points = load_aq(start_date, end_date)
    
    print("Aggregating Airbeam data\n")
    aq_by_block, aq_by_neighborhood, aq_by_hexagon, aq_by_big_hexagon = process_data.process_aq(aq_points, blocks, neighborhoods, hexagons, big_hexagons)
    print("Saving Airbeam data\n")
    
    current_aq_by_block = current_aq_by_block.append(aq_by_block)
    current_aq_by_block.to_csv(os.path.join(OUTDIR, 'aq_by_block.csv'), index=False)
    
    current_aq_by_neighborhood = current_aq_by_neighborhood.append(aq_by_neighborhood)
    current_aq_by_neighborhood.to_csv(os.path.join(OUTDIR, 'aq_by_neighborhood.csv'), index=False)
    
    current_aq_by_hexagon = current_aq_by_hexagon.append(aq_by_hexagon)
    current_aq_by_hexagon.to_csv(os.path.join(OUTDIR, 'aq_by_hexagon.csv'), index=False)
    
    current_aq_by_big_hexagon = current_aq_by_big_hexagon.append(aq_by_big_hexagon)
    current_aq_by_big_hexagon.to_csv(os.path.join(OUTDIR, 'aq_by_big_hexagon.csv'), index=False)
    
    print("Processing Airbeam neighborhood data\n")
    neigh_points = gpd.GeoDataFrame(aq_points, geometry=gpd.points_from_xy(aq_points.lng, aq_points.lat))
    neigh_points.crs = "EPSG:4326"
    neigh_points.to_crs(4326)
    neigh_points = gpd.sjoin(neigh_points, blocks[['geoid10', 'geometry']], how="left", op='intersects')
    neigh_points = neigh_points.drop(columns=['index_right'])
    neigh_points = gpd.sjoin(neigh_points, neighborhoods, how="left", op='intersects')
    neigh_points = neigh_points.drop(columns=['index_right'])

    neighs = list(neighborhoods['pri_neigh'].unique())
    for n in neighs:
        print('neighborhood: ', n)
        neigh_tmp = create_neighborhood_data(neigh_points, n)
        if len(neigh_tmp) > 0:
            print('there is new data')
            if os.path.isfile(os.path.join(NEIGH_DIR, n + '.csv')):
                print('append to existing file')
                current_neigh_points = pd.read_csv(os.path.join(NEIGH_DIR, n + '.csv'))
                pipeline.convert_to_timeseries(current_neigh_points, ['thirtymins'])
                current_neigh_points = truncate_dates(current_neigh_points, 'thirtymins', last_updated)
                os.rename(os.path.join(NEIGH_DIR, n + '.csv'), os.path.join(OUTDIR, 'old', n + '.csv'))
                current_neigh_points = current_neigh_points.append(neigh_tmp)
                current_neigh_points.to_csv(os.path.join(NEIGH_DIR, n + '.csv'), index=False)
            else:
                print('create new file')
                neigh_tmp.to_csv(os.path.join(NEIGH_DIR, n + '.csv'), index=False)

