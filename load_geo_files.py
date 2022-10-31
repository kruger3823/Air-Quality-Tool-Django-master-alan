'''
Retrieve relevant geo/shape files for Chicago.
Store this data as .csv files in './data'.
'''

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


DATA_DIR = os.path.dirname(__file__)
OUTDIR = os.path.join(DATA_DIR, 'data')
INDIR = os.path.join(OUTDIR, 'downloaded')
NEIGH_DIR = os.path.join(OUTDIR, 'neighs')
TEMP_DIR = '/Users/ldinh/Documents/GitHub/Air-Quality-Tool/data/points/'


def load_blocks():

    blocks = gpd.read_file("https://data.cityofchicago.org/resource/bt9m-d2mf.geojson?$limit=9999999")
    blocks.crs = "EPSG:4326"
    blocks["geo_12"] = blocks["geoid10"].map(lambda x: str(x)[:12])
    blocks = blocks.drop_duplicates()
    blocks['area'] = blocks.to_crs('EPSG:3857').area / (10**6)
    blocks = blocks[['geo_12', 'geoid10', 'geometry', 'area']]

    return blocks


def load_neighborhoods():

    neighborhoods = gpd.read_file("https://data.cityofchicago.org/resource/y6yq-dbs2.geojson?$limit=9999999")
    neighborhoods.crs = "EPSG:4326"
    neighborhoods['area'] = neighborhoods.to_crs('EPSG:3857').area / (10**6)

    return neighborhoods


def load_hexagons():

    hexagons = pd.read_csv(os.path.join(INDIR, 'download.csv'))
    hexagons = hexagons.rename(columns={'geo': 'geometry'})
    hexagons['geometry'] = hexagons['geometry'].apply(wkt.loads)
    hexagons = gpd.GeoDataFrame(hexagons[['id', 'geometry']], geometry='geometry')
    hexagons.crs = "EPSG:4326"

    return hexagons


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
    #print(f"Distance: {meters} m")
    #print(f"Distance: {km} km")
    #print(f"Distance: {miles} miles")
    return miles


def load_big_hexagons():

    boundary = gpd.read_file("https://data.cityofchicago.org/resource/y6yq-dbs2.geojson?$limit=9999999")[['geometry']]
    boundary['temp'] = 1
    boundary = boundary.dissolve(by='temp')
    boundary = boundary[['geometry']]
    boundary.crs = "EPSG:4326"

    xmin, ymin, xmax, ymax = boundary.total_bounds # lat-long of 2 corners
    # East-West extent of Chicago = 21 miles
    EW = haversine((xmin, ymin), (xmax, ymin))
    # North-South extent of Chicago = 26 miles
    NS = haversine((xmin, ymin), (xmin, ymax))
    # diamter of each hexagon in the grid = 0.5 miles
    d = 1
    # horizontal width of hexagon = w = d*sin(60)
    w = d*np.sin(np.pi/3)
    # Approximate number of hexagons per row = EW/w 
    n_cols = int(EW/w)+1
    # Approximate number of hexagons per column = NS/d
    n_rows = int(NS/d)+ 1

    w = (xmax-xmin)/n_cols # width of hexagon
    d = w/np.sin(np.pi/3) # diameter of hexagon
    array_of_hexes = []

    for rows in range(0,n_rows):
        hcoord = np.arange(xmin,xmax,w) + (rows%2)*w/2
        vcoord = [ymax- rows*d*0.75]*n_cols
        for x, y in zip(hcoord, vcoord):#, colors):
            hexes = RegularPolygon((x, y), numVertices=6, radius=d/2, alpha=0.2, edgecolor='k')
            verts = hexes.get_path().vertices
            trans = hexes.get_patch_transform()
            points = trans.transform(verts)
            array_of_hexes.append(Polygon(points))

    big_hexagons = gpd.GeoDataFrame({'geometry': array_of_hexes}) #crs={'init': 'EPSG:4326'}
    big_hexagons.crs = "EPSG:4326"
    big_hexagons = big_hexagons.reset_index().rename(columns={'index': 'id'})
    
    return big_hexagons
       

if __name__ == "__main__":
    blocks = load_blocks()
    blocks.to_csv(os.path.join(OUTDIR, 'blocks.csv'), index=False)
    neighborhoods = load_neighborhoods()
    neighborhoods.to_csv(os.path.join(OUTDIR, 'neighborhoods.csv'), index=False)
    hexagons = load_hexagons()
    hexagons.to_csv(os.path.join(OUTDIR, 'hexagons.csv'), index=False)
    big_hexagons = load_big_hexagons()
    big_hexagons.to_csv(os.path.join(OUTDIR, 'big_hexagons.csv'), index=False)

    