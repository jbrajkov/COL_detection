#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:07:27 2026

@author: josip
"""


"""
Example script to run the COL detection algorithm on a given dataset.

You can visualize the algorithm running and modify its parameters by using the sample datasets provided
(sample_daily_precip.nc and sampe_daily_z500.nc)
These datasets cover the period 2019-10-20/2019-10-25

More precisely, this script demonstrates how to:
- Define input datasets (z500 and precipitation)
- Configure detection parameters
- Run the COL detection algorithm (COL_research)
- Save outputs and clean memory

The script supports:
- ERA5 reanalysis
- CMIP6 climate models (historical + scenarios)

Notes
-----
- Paths must be adapted to the user environment
- Parameters such as thresholds, spatial domain, and percentiles
  strongly influence the detection results
- This script is intended as an example and can be customized
  for specific studies

Main parameters
---------------
mo : str
    Model name (e.g., 'ERA5', 'MIROC6')

ts : int
    Temporal resolution (in hours)

thp : float
    Precipitation threshold used for event selection

cta : float
    Minimum COL area (km²)

qup : float
    Percentile used for precipitation thresholding

qu_rain : float
    Percentile used to define if a pixel is considered as a precipitation pixel


test_algo : int (0 or 1)
    If 1, runs a single test case with plotting enabled
"""

from COL_detection import col_detection

mo = 'ERA5'
ts = 24

thp = 0
cta = 50000
qup = 99.9
qu_rain=95


test_algo = 1

"string to format output text files names"

grid = '_ERA5grid'
add_message = ''

    
string_extension = f"_qu{qup:.0f}_area_{cta:.0f}km2_{ts:.0f}h_{thp:.0f}mm{grid}{add_message}"

mod='ERA5'
  
dire = '/home/josip/Desktop/scripts/test_algo/'
dir_out='/home/josip/Desktop/'
fnm=f'{dire}ERA5_mask_LS.nc'
fn_topo=f'{dire}ERA5_topo.nc'
    
  
fns = [dire+'sample_daily_z500.nc']



ec = col_detection(fns=fns,
                 varnames=['z500'],
                 fnp=dire+'sample_daily_precip.nc',
                 dir_out="/home/josip/",
                 ave_quant=1,
                 qu=qup,
                 topo_mask=0,
                 fn_mask=fnm,
                 fnto=fn_topo,
                 interpolate_z500=1)

ec.COL_research(plt_or_not=test_algo, mod=mod, ts=ts, thp=thp, thf=thp, dirout=dir_out,
                critere_taille=cta, ste=10, numero=string_extension,
                lat_max=60,
                lat_min=35,
                lon_max=35,
                lon_min=-20,
                test_mode=test_algo,
                d2p=None,  
                criteria=qu_rain,                        
                )
ec.close()
del ec

