#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load and plot GHG data from the LRHO watershed.

For each site, this script will load in the sensor discharge data, along with fDOM (and etc),
and plot the cumulative discharge and variable for each year.

Parameters
----------
info

subfolder

LMPghg_input

QUESTghg_input

grab_input

"""
"""
Created on Thu Oct 16 14:21:08 2025

@author: etipps
"""

#%% Project Title 
"""
 Scaling Greenhouse Gases through a Small Watershed
"""

#%% Introduction 
"""

"""

#%% Research question and hypotheses
"""

"""

#%% Study site
"""

"""

#%% Data sets
"""

"""

#%% Imports
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import matplotlib.cm as cm

#%% Parameters
info = 'sites.txt'

subfolder = Path('Data')

LMPghg_input = '251014 EPSCoR Gas Query.xlsx'

QUESTghg_input = '251014 QuEST Lamprey Data with GHG.xlsx'

grab_input = '20251001 Lamprey Weekly Monthly QAQC w formatting.xlsx'

Q_lower = 'usgs_newmarket.txt'

Q_upper = 'usgs_raymond.txt'

#%% Generate site list
with open(info, "r") as sitenames:
	sites = sitenames.readlines()
    
sites = sites[0]
sites=sites.split(',')
#%% Function to read in data

def wrangle(datafile, sites):
    """"
    Read in excel file of Lamprey data, set collection date as index and remove measurements without GHG
    data
    
    Parameters
    ----------
    datafile : string
        Name of filename to be read in
        
    Returns
    ---------
    data : DataFrame
        Wrangled data frame
    """
    
    filename = subfolder/datafile

    data = pd.read_excel(filename,
                           parse_dates=['Collection Date'], index_col='Collection Date')
    
    # Remove sites that aren't in site list
    data = data[data['Sample Name'].isin(sites)]
    
    # Loop through columns to remove rows with GHG data 
    for col in data.columns.tolist():
        if col.startswith('CH4') or col.startswith('CO2') or col.startswith('N2O'):
            data.dropna(subset = col, inplace = True)
    
    # sort df by index
    data.sort_index(inplace=True)
    
    return data

#%% Wrangle input data
quest_data = wrangle(QUESTghg_input, sites)

LMP_data = wrangle(LMPghg_input, sites)

grab_data = wrangle(grab_input, sites)  

# splice LMP_data to have the same start time as quest
start = quest_data.index[0]
LMP_data = LMP_data[start:]   

#%% Import and wrangle discharge data
upper_q = pd.read_csv(subfolder/Q_upper, delimiter = '\t', comment = '#', header = 1,
                     parse_dates=['20d'], index_col=['20d'])

lower_q = pd.read_csv(subfolder/Q_lower, delimiter = '\t', comment = '#', header = 1,
                     parse_dates=['20d'], index_col=['20d'])

# Rename discharge columns and drop irrelevant columns
upper_q.rename(columns = {"14n":"Q_CFS"}, inplace = True)
upper_q.index.names = ['DATE']

lower_q.rename(columns = {"14n":"Q_CFS"}, inplace = True)
lower_q.index.names = ['DATE']

#%% Initial timeseries
colors = cm.get_cmap('tab10', len(sites))
fig_title = 'Lamprey GHG timeseries'
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize = (10,9), sharex = True)
fig.suptitle(fig_title)

# Plot 1: CO2 from each site
for idx, site in enumerate(sites):
    site_data = quest_data[quest_data['Sample Name'] == site]
    ax1.scatter(site_data.index, site_data['CO2_uM'],
                label=site, color=colors(idx))

for idx, site in enumerate(sites):
    site_data = LMP_data[LMP_data['Sample Name'] == site]
    ax1.scatter(site_data.index, site_data['CO2_umol_L'],
                label=None, color=colors(idx))

ax1.set_ylabel('CO2 (uM)')

# Plot 2: CH4 from each site
for idx, site in enumerate(sites):
    site_data = quest_data[quest_data['Sample Name'] == site]
    ax2.scatter(site_data.index, site_data['CH4_uM'],
                label=site, color=colors(idx))

for idx, site in enumerate(sites):
    site_data = LMP_data[LMP_data['Sample Name'] == site]
    ax2.scatter(site_data.index, site_data['CH4_umol_L'],
                label=None, color=colors(idx))

ax2.set_ylabel('CH4 (uM)')

# Plot 3: N2O from each site
for idx, site in enumerate(sites):
    site_data = quest_data[quest_data['Sample Name'] == site]
    ax3.scatter(site_data.index, site_data['N2O_uM'],
                label=site, color=colors(idx))

for idx, site in enumerate(sites):
    site_data = LMP_data[LMP_data['Sample Name'] == site]
    ax3.scatter(site_data.index, site_data['N2O_umol_L'],
                label=None, color=colors(idx))

ax3.set_ylabel('N2O (uM)')

# Plot 4: Discharge timeseries
ax5 = ax4.twinx()
ax4.plot(upper_q["Q_CFS"], color = "turquoise")
ax5.plot(lower_q["Q_CFS"], color = "blue")

ax4.set_ylabel('Upper catchment discharge (cfs)', color = "turquoise")
ax5.set_ylabel('Lower catchment discharge (cfs)', color = "blue")

ax4.set_xlabel('Time')

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels,
           loc='lower center', ncol=len(sites), bbox_to_anchor=(0.5, -0.02),
           title='Sites')

plt.tight_layout(rect=[0, 0.05, 1, 1])  
plt.show()

# Planned adjustments to timeseries plot:

#%% CQ slope of Greenhouse gases along Lamprey watershed
"""

"""

#%% CQ slope v. CVc/CVq
"""

"""

#%% Greenhouse gases with biological indicators (NH4:NO3-)
"""

"""

#%% Discussion/conclusion
"""

"""


