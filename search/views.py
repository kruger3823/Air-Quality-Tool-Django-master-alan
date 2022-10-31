from django.shortcuts import render, HttpResponse
from django import forms
import json
import os
import csv
from plotly.offline import plot
import urllib
import requests
#import plotly.plotly as py


#------------------------------------------signup-------------------->

from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from .forms import UserRegisterForm
from django.core.mail import send_mail
from django.core.mail import EmailMultiAlternatives
from django.template.loader import get_template
from django.template import Context


#################### index#######################################
def index(request):
	return render(request, 'user/indexmain.html', {'title':'index'})

########### register here #####################################
def register(request):
	if request.method == 'POST':
		form = UserRegisterForm(request.POST)
		if form.is_valid():
			form.save()
			return redirect('login')
	else:
		form = UserRegisterForm()
	return render(request, 'register.html', {'form': form, 'title':'register here'})

################ login forms###################################################
def Login(request):
	if request.method == 'POST':

		# AuthenticationForm_can_also_be_used__

		username = request.POST['username']
		password = request.POST['password']
		user = authenticate(request, username = username, password = password)
		if user is not None:
			form = login(request, user)
			messages.success(request, f' welcome {username} !!')
			return redirect('home')
		else:
			messages.info(request, f'account done not exist please sign in')
	form = AuthenticationForm()
	return render(request, 'login.html', {'form':form, 'title':'log in'})







DATA_DIR = os.path.dirname(__file__)
STATIC = os.path.join(DATA_DIR, '../static')
URL_STATIC = 'https://chicagoairquality.s3.us-east-2.amazonaws.com/static/'
MANUAL_DIR = os.path.join(DATA_DIR, '../data', 'manually_created')


def gen_dropdown(filename):
    '''
    Generates drop down list from a file (e.g. csv).

    Inputs:
        - filename (string): name of file to use as input to drop down
    Outputs:
        - list of drop down options
    '''
    with open(os.path.join(MANUAL_DIR, filename)) as f:
        f.readline()
        col = sorted([tuple(line) for line in csv.reader(f)])
        col.insert(0, ("None", "None"))

    return [x for x in col]


NEIGHS = gen_dropdown('neighs.csv')
YEARS = gen_dropdown('years.csv')


class ChartForm(forms.Form):
    '''
    Creates a class to contain forms for user input.
    '''
    all_years = forms.BooleanField(label='Show summary of all years',
        required=False, initial=True)

    year = forms.ChoiceField(label='Select year to drilldown', 
        choices=YEARS, required=False, initial='2020')
    two_miles = forms.BooleanField(label='Only include AQ readings within 2 mi of an EPA/PA sensor',
        required=False)
    heatmaps = forms.BooleanField(label='Show heatmaps',
        required=False, initial=True)
    timeseries = forms.BooleanField(label='Show timeseries',
        required=False, initial=True)
    scatters = forms.BooleanField(label='Show scatters',
        required=False, initial=True)
    boxplots = forms.BooleanField(label='Show boxplots',
        required=False, initial=True)

    month = forms.ChoiceField(label='Select month to drilldown', 
        choices=[(4, 'Apr'), (5, 'May'), (6, 'Jun'), (7, 'Jul'), (8, 'Aug'), (9, 'Sep')], required=False)

    neigh = forms.ChoiceField(label='Select neighborhood to drilldown', 
        choices=NEIGHS, required=False, initial='Hyde Park')


def home(request):
    '''
    Generates an instance of a StateForm on webpage. After user enters
    inputs into form, uses selections to generate tables and graphs.

    Inputs:
        - request: 

    Outputs:
        - renders State Form
    '''
    context = {}
    res = None
    # if this is a POST request we need to process the form date
    if request.method == 'GET':
        # create a form instance and populate it with data from the request:
        if len(request.GET):
            form = ChartForm(request.GET)
        else:
            form = ChartForm()
        # check whether it's valid:
        if form.is_valid():
            args = {}
            if form.cleaned_data['year']:
                args['year'] = form.cleaned_data['year']

                context['two_miles'] = False
                if form.cleaned_data['two_miles']:
                    context['two_miles'] = form.cleaned_data['two_miles']

                context['heatmaps'] = False
                if form.cleaned_data['heatmaps']:
                    context['heatmaps'] = form.cleaned_data['heatmaps']

                context['timeseries'] = False
                if form.cleaned_data['timeseries']:
                    context['timeseries'] = form.cleaned_data['timeseries']

                context['scatters'] = False
                if form.cleaned_data['scatters']:
                    context['scatters'] = form.cleaned_data['scatters']

                context['boxplots'] = False
                if form.cleaned_data['boxplots']:
                    context['boxplots'] = form.cleaned_data['boxplots']

                context['all_years'] = False
                if form.cleaned_data['all_years']:
                    context['all_years'] = form.cleaned_data['all_years']

                if form.cleaned_data['month']:
                    args['month'] = form.cleaned_data['month']
                
            if form.cleaned_data['neigh'] and form.cleaned_data['neigh'] != 'None':
                args['neigh'] = form.cleaned_data['neigh']

            if args:
                if not context['two_miles']:
                    if context['heatmaps'] and args['year'] != 'None':
                        context['heatmaps'] = [''.join((URL_STATIC, 'graphs/aq/Heatmap_Daily_Avg_PM25_Summer_', args['year'], '.png'))]
                        context['heatmaps'].append(''.join((URL_STATIC, 'graphs/aq/Heatmap_Daily_Harmful_PM25_Summer_', args['year'], '.png')))
                        context['heatmaps'].append(''.join((URL_STATIC, 'graphs/aq/Heatmap_Daily_Low_PM25_Summer_', args['year'], '.png')))
                    else:
                        context['heatmaps'] = False

                    if context['scatters'] and args['year'] != 'None': 
                        context['scatters'] = [''.join((URL_STATIC, 'graphs/aq/Scatter_Daily_Avg_PM25_AQ_mean_vs_EPA_mean_by_', args['year'], '.png'))]
                        context['scatters'].append(''.join((URL_STATIC, 'graphs/aq/Scatter_Daily_Avg_PM25_AQ_mean_vs_PA_mean_by_', args['year'], '.png')))
                        if context['boxplots']:
                            context['scatters'].append(''.join((URL_STATIC, 'graphs/aq/Boxplot_Daily_Avg_PM25_Summer_', args['year'], '.png')))
                    else:
                        context['scatters'] = False

                    if context['timeseries'] and args['year'] != 'None':
                        context['timeseries'] = [''.join((URL_STATIC, 'graphs/aq/Timeseries_Daily_Avg_PM25_Summer_', args['year'], '.png'))]
                    else:
                        context['timeseries'] = False

                    if context['all_years'] or args['year'] == 'None':
                        context['all_years0'] = [''.join((URL_STATIC, 'graphs/neighborhood_summary_Daily', '.png'))]
                        context['all_years'] = [''.join((URL_STATIC, 'graphs/aq/Scatter_Daily_Avg_PM25_AQ_mean_vs_EPA_mean_by_Year', '.png'))]
                        context['all_years'].append(''.join((URL_STATIC, 'graphs/aq/Scatter_Daily_Avg_PM25_AQ_mean_vs_PA_mean_by_Year', '.png')))
                        context['all_years'].append(''.join((URL_STATIC, 'graphs/aq/Boxplot_Daily_Avg_PM25_All_Summers', '.png')))

                else:
                    if context['heatmaps'] and args['year'] != 'None':
                        context['heatmaps'] = [''.join((URL_STATIC, 'graphs/aq_within_2_miles_epa/Heatmap_Daily_Avg_PM25_Summer_', args['year'], '.png'))]
                        context['heatmaps'].append(''.join((URL_STATIC, 'graphs/aq_within_2_miles_epa/Heatmap_Daily_Harmful_PM25_Summer_', args['year'], '.png')))
                        context['heatmaps'].append(''.join((URL_STATIC, 'graphs/aq_within_2_miles_epa/Heatmap_Daily_Low_PM25_Summer_', args['year'], '.png')))
                        context['heatmaps1'] = [''.join((URL_STATIC, 'graphs/aq_within_2_miles_pa/Heatmap_Daily_Avg_PM25_Summer_', args['year'], '.png'))]
                        context['heatmaps1'].append(''.join((URL_STATIC, 'graphs/aq_within_2_miles_pa/Heatmap_Daily_Harmful_PM25_Summer_', args['year'], '.png')))
                        context['heatmaps1'].append(''.join((URL_STATIC, 'graphs/aq_within_2_miles_pa/Heatmap_Daily_Low_PM25_Summer_', args['year'], '.png')))
                    else:
                        context['heatmaps'] = False
                        context['heatmaps1'] = False

                    if context['scatters'] and args['year'] != 'None': 
                        context['scatters'] = [''.join((URL_STATIC, 'graphs/aq_within_2_miles_epa/Scatter_Daily_Avg_PM25_AQ_mean_vs_EPA_mean_by_', args['year'], '.png'))]
                        if context['boxplots'] and args['year'] != 'None':
                            context['scatters'].append(''.join((URL_STATIC, 'graphs/aq_within_2_miles_epa/Boxplot_Daily_Avg_PM25_Summer_', args['year'], '.png')))
                        context['scatters'].append(''.join((URL_STATIC, 'graphs/aq_within_2_miles_pa/Scatter_Daily_Avg_PM25_AQ_mean_vs_PA_mean_by_', args['year'], '.png')))
                        if context['boxplots'] and args['year'] != 'None':
                            context['scatters'].append(''.join((URL_STATIC, 'graphs/aq_within_2_miles_pa/Boxplot_Daily_Avg_PM25_Summer_', args['year'], '.png')))
                    else:
                        context['scatters'] = False

                    if context['timeseries'] and args['year'] != 'None':
                        context['timeseries'] = [''.join((URL_STATIC, 'graphs/aq_within_2_miles_epa/Timeseries_Daily_Avg_PM25_Summer_', args['year'], '.png'))]
                        context['timeseries'].append(''.join((URL_STATIC, 'graphs/aq_within_2_miles_pa/Timeseries_Daily_Avg_PM25_Summer_', args['year'], '.png')))
                    else:
                        context['timeseries'] = False

                    if context['all_years'] or args['year'] == 'None':
                        context['all_years0'] = [''.join((URL_STATIC, 'graphs/neighborhood_summary_Daily', '.png'))]
                        context['all_years'] = [''.join((URL_STATIC, 'graphs/aq_within_2_miles_epa/Scatter_Daily_Avg_PM25_AQ_mean_vs_EPA_mean_by_Year', '.png'))]
                        context['all_years'].append(''.join((URL_STATIC, 'graphs/aq_within_2_miles_epa/Boxplot_Daily_Avg_PM25_All_Summers', '.png')))
                        context['all_years'].append(''.join((URL_STATIC, 'graphs/aq_within_2_miles_pa/Scatter_Daily_Avg_PM25_AQ_mean_vs_PA_mean_by_Year', '.png')))
                        context['all_years'].append(''.join((URL_STATIC, 'graphs/aq_within_2_miles_pa/Boxplot_Daily_Avg_PM25_All_Summers', '.png')))
                
                if args.get('year', None) and args.get('month', None):
                    m = args.get('month', None)
                    y = args.get('year', None)

                    context['month_plot'] = []
                    f = requests.get(''.join((URL_STATIC, 'graphs/aq/comparison_daily_maps_month_', str(m), '_year_', str(y), '.html')))
                    context['month_plot'].append(f.text)

                if args.get('neigh', None):
                    n = args.get('neigh', None)

                    context['neigh_plot'] = []
                    for v in ['yearly', 'monthyear']:
                        context['neigh_plot'].append(''.join((URL_STATIC, 'graphs/neighs/', n, '_timeseries_', v, '.png')))

                    context['plot_div0'] = []
                    for v0 in ['yearly', 'monthyear']:
                        f = requests.get(''.join((URL_STATIC, 'graphs/neighs/', n, '_block_', v0, '.html')))
                        context['plot_div0'].append(f.text)

                    context['plot_div'] = []
                    for v1 in ['monthly', 'hourly']:
                        f = requests.get(''.join((URL_STATIC, 'graphs/neighs/', n, '_lon_lat_', v1, '.html')))
                        context['plot_div'].append(f.text)

    else:
        form = ChartForm()

    context['form'] = form

    return render(request, 'indexmain.html', context)






