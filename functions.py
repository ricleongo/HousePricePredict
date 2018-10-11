import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
from functools import reduce

def calculate_percentage(value, total):
    return (value / total) * 100

def get_zillow_column_des() :
    return {
        "2010": "Zillow Median Value Housing 2010",
        "2011": "Zillow Median Value Housing 2011",
        "2012": "Zillow Median Value Housing 2012",
        "2013": "Zillow Median Value Housing 2013",
        "2014": "Zillow Median Value Housing 2014",
        "2015": "Zillow Median Value Housing 2015",
        "2016": "Zillow Median Value Housing 2016",
        "2017": "Zillow Median Value Housing 2017",
        "2018": "Zillow Median Value Housing 2018"
    }  

# funtion that bring the column names per year.
def get_column_des(year) :
    return {
        "NAME": "Name",
        "state": "State",
        "county": "MunicipalCodeFIPS",
        "B01003_001E": "Population {}".format(year),
        "B01002_001E": "Median Age {}".format(year),
        "B19013_001E": "Household Income {}".format(year),
        "B19301_001E": "Per Capita Income {}".format(year),
        "B25077_001E": "Median Value Housing Units {}".format(year),
        "B17001_002E": "Poverty Count {}".format(year)
        # "B23025_005E": "Unemployment Count {}".format(year),
    }

# Merge method on all years.
def get_merged_data(array_data, key_fields):
    return reduce(lambda left, right: 
                  pd.merge(left, right, 
                           on=key_fields,
                           how='outer'), 
                  array_data).fillna('void')
    
# Fields to be retrieved from census API
def get_fields():
    return ("NAME",
          "B01003_001E", 
          "B01002_001E", 
          "B19013_001E", 
          "B19301_001E", 
          "B25077_001E", 
          "B17001_002E")

# Filter by County and State
def get_filters():
     return {
                'for': 'county: 121,135,089,067,063,057,151,117,097,077,045,113,015,297,013,227,085,199,171', 
                'in': 'state: 13'
            }


def print_message(message):
    print("-----------------------------------------")
    print(message)
    print("-----------------------------------------")


def print_plot(data_set, x_values, title, xlabel, ylable, limX):

    fig, ax = plt.subplots()

    cm = plt.get_cmap('tab20')
    ax.set_color_cycle([cm(1.*i/20) for i in range(20)])

    for i in range(len(data_set)):
        value = data_set.iloc[i,:]
        ax.plot(x_values, value, label=data_set.index[i][1], linewidth = 4)
        ax.scatter(x_values, value, label=None, s = 80)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylable)

    ax.set_xlim(left=limX[0], right=limX[1])
    ax.legend(loc="upper right")
    ax.grid()

    fig.set_size_inches(16, 10)
    fig.show(warn=False)

def print_bars(data_set, x_values, title, xlabel, ylable, limX):
    fig, ax = plt.subplots()

    cm = plt.get_cmap('tab20')
    ax.set_color_cycle([cm(1.*i/20) for i in range(20)])


    for i in range(len(data_set.index)):
        value = data_set.iloc[i,:]    
        ax.bar(x_values, value, label=data_set.index[i][1])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylable)

    ax.set_xlim(left=limX[0], right=limX[1])
    ax.legend(loc="upper right")
    ax.grid()

    fig.set_size_inches(16, 10)
    fig.show(warn=False)    

def print_trendline(x, y):
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x,  p(x), ":")

def get_std_error(data, div):

    lim = len(data) // div

    samples = [data.iloc[(i * div):(i * div + div)] for i in range(0, lim)]

    means = [np.mean(s) for s in samples]
    standard_errors = [sem(s) for s in samples]

    return np.arange(0, len(means)), means, standard_errors, div
