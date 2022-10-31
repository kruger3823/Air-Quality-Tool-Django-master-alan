#!/bin/bash

if [ $1 = "-d" ] ; then
echo "Loading geo/shape files ..."
python3 load_geo_files.py

echo "API-ing ..."
python3 load_data_files.py
fi

if [ $1 = "-g" ] ; then 
echo "Generating visualizations"
python3 create_visualizations.py
fi

echo "Pulling up UI"
python3 manage.py runserver