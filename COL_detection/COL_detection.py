#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:36:07 2024

@author: josip
"""

"""
Cut-Off Low (COL) detection algorithm based on z500 fields.

Main class:
    EOF_climate

Main method:
    col()

Description:
    Detects enclosed low geopotential structures (COLs) and attributes
    precipitation events based on spatial proximity.

Author: Josip
"""

import xarray as xr
import commontools_zenodo as ct
import numpy as np
import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy
import os
from scipy.interpolate import CubicSpline
from datetime import datetime

import gc


class col_detection():

    def __init__(self, fns=['~/Desktop/scripts/Netcdfs/ERA5_cl10mm_z500_1951-2021.nc'], # NetCDF file(s) containing geopotential height (z500)
                 varnames=['z500'],                                                     # Variable name for z500 in the NetCDF files
                 fnp='~/Desktop/scripts/Netcdfs/Precip_ERA5_cl10mm_z500_1951-2021.nc',  # NetCDF file containing associated precipitation
                 dir_out="/home/josip//Desktop/analogues1/",                            # Output directory for figures
                 ave_quant=0,                                                           # Defines precipitation time series: mean (0) or percentile (1)
                 qu=90,                                                                 # Percentile value if ave_quant != 0
                 fn_mask='/srv7_tmp1/jbrajkovic/These/SPATIAL/Belgium_pix.nc4',         # Spatial mask file for precipitation domain
                 
                 lonm1=-10, lonm2=30, latm1=36, latm2=57,                               # Longitude/latitude bounds for precipitation domain
                 fnto='',                                                               # File containing topography
                 topo_mask=0,                                                           # Whether to apply topographic filtering
                 interpolate_z500=0                                                     # Whether to interpolate z500 onto ERA5 grid
                 ):
        
        "Definition of all object variables"
        
        self.fns = fns
        self.fnp = fnp
        
        "Information used for plotting (models, colors, scenarios)"
        
        self.models = np.array(
            ['MPI-ESM1', 'EC3', 'MIROC6', 'NorESM2', 'CMCC-CM2-SR5', 'IPSL'])
        self.colors = ['darkblue', 'orange', 'orangered', 'darkred']
        self.scens = np.array(['126', '245', '370', '585'])
        
        self.varnames = varnames
        self.dir_out = dir_out
   
        
        print('Synoptic file names:', fns)
        
        # Read longitude and latitude grids from z500 files
        self.lons, self.lats = self.read_grid(fn=self.fns[0])
        self.lons_nomodif = np.copy(self.lons)
        self.lats_nomodif = np.copy(self.lats)
        
        # Resolution ratio relative to a reference 181-lat grid
        self.rap_taille = int(181 / self.lats.shape[0])
        
        # Read precipitation grid
        self.lons1, self.lats1 = self.read_grid(fn=fnp)
        self.fn_mask = fn_mask
        
     
        self.fn_coord_era = self.fn_mask
        self.lons_ERA, self.lats_ERA = self.read_grid(fn=self.fn_coord_era)
        
        
        "Creation of the precipitation mask (European land pixels within a defined region)"
        
        if 'ERA' in fn_mask:
            
            # Load land-sea mask (ERA format)
            self.maskp = np.array(xr.open_dataset(fn_mask)['MSK']) == 1
            self.eralm = np.copy(self.maskp)
            self.mask_map = np.array(self.maskp)
            
            print('shape', self.maskp.shape, self.lons1.shape)
            
            # Load topography and convert geopotential to meters
            topo = xr.open_dataset(fnto)
            self.ztop = np.transpose(np.array(topo['geop']) / 9.81)[:, :, 0]
            
            # Optional topographic filtering (e.g., exclude high elevations)
            if topo_mask:
                mtop = self.ztop < 300
            else:
                mtop = np.transpose(np.ones_like(self.mask_map))
            
            # Final mask combining spatial domain, land mask, and topography
            self.maskp = (
                (self.lons1 > lonm1) & (self.lons1 < lonm2) &
                (self.lats1 > latm1) & (self.lats1 < latm2) &
                (np.transpose(self.maskp)) &
                (mtop)
            )

        else:
            # Non-ERA mask handling
            self.maskp = np.array(xr.open_dataset(fn_mask)['MSK']) == 1
            
            # Ensure mask orientation matches precipitation grid
            if self.maskp.shape[0] != self.lons1.shape[0]:
                self.maskp = self.maskp.T
                
            self.mask_map = self.maskp
            self.ztop = np.ones_like(self.mask_map)
            
        self.ave_quant = ave_quant
        self.qu = qu
        
        # Load precipitation dataset
        dsp = xr.open_dataset(self.fnp)
        self.dsp = dsp
        maskp = self.maskp
        
        
        "Reading and processing daily precipitation time series"
        
        # Define cache file to avoid recomputation
        cache_dir = "/srv5_tmp2/jbrajkovic/"
        base = os.path.splitext(os.path.basename(self.fnp))[0]
        cache_file = os.path.join(cache_dir, f"{base}_pr_ts.npy")
        
        if os.path.isfile(cache_file):
            print("Reading cached pr_ts:", cache_file)
            self.pr_ts = np.load(cache_file)

        else:
            # Load precipitation data
            self.datap = np.asarray(dsp['Precip'])
            datap = self.datap
            
            # Compute time series: mean or percentile over masked region
            if self.ave_quant == 0:
                self.pr_ts = np.nanmean(datap[:, maskp], axis=1)
            else:
                self.pr_ts = np.nanquantile(
                    datap[:, maskp],
                    self.qu * 0.01,
                    axis=1
                )
            
            # Save computed time series to cache
            try:
                np.save(cache_file, self.pr_ts)
                del self.datap
            except:
                print('Cache file could not be created')
        
        # Load z500 dataset and time axis
        self.ds = xr.open_dataset(self.fns[0])
        self.time = pd.to_datetime(np.array(self.ds['time']))
        
        print('Dataset start and end dates:', self.time[0], self.time[-1])
            
        self.ds_data = xr.open_dataset(fns[0])
        self.interpolate_z500 = interpolate_z500

        # Marker styles for plotting
        self.markers_lines = ['*--', '^--', 's--', 'D--', '<--', '>--']
        self.markers = ['*', '^', 's', 'D', '<', '>']

        # Compute latitudinal resolution of z500 grid
        self.res_lat = abs(self.lats[0, 0] - self.lats[1, 0])
        
        if self.res_lat == 0:
            self.res_lat = abs(self.lats[0, 0] - self.lats[0, 1])

        # Compute pixel area (approximate, in km²)
        self.pix_area = np.ones_like(self.lons)
        self.pix_area[:, :] = (
            self.res_lat * np.cos(np.deg2rad(self.lats))
        ) * 111 * (self.res_lat * 111)
        
        "Mask defining the spatial domain used for z500 histograms"
        
        self.lon_etendue1 = -35
        self.lon_etendue2 = 46
        self.lat_etendue1 = 30
        self.lat_etendue2 = 70
        
        self.mask_sp = (
            (self.lons > self.lon_etendue1) & (self.lons < self.lon_etendue2) &
            (self.lats > self.lat_etendue1) & (self.lats < self.lat_etendue2)
        )
        
        self.dsp = dsp
        self.Pr = None
    
    def read_grid(self, fn=''):
        # fn=self.fns[0]
        print(fn)
        ds = xr.open_dataset(fn)
        print(ds.keys())
        lo, la = np.array(ds['LON']), np.array(ds['LAT'])
        return(lo, la)
    
    def read_data(self):
        "Reads the z500 (and possibly other variables) and returns a data array with time as the first dimension"

        # Open all required datasets once (avoid repeated I/O operations)
        dsets = [xr.open_dataset(f) for f in self.fns[:self.ndim]]
    
        # Store variables as NumPy arrays
        vars_np = []
    
        for i, ds in enumerate(dsets):
            var = ds[self.varnames[i]]
    
            # Apply unit conversions when necessary
            # ERA5 geopotential is converted to meters
            if 'ERA5_' in self.fns[i]:
                var = var / 9.81
            
            # Mean sea level pressure converted from Pa to hPa
            if self.varnames[i] == 'msl':
                var = var / 100
    
            # Convert xarray DataArray to NumPy array
            vars_np.append(np.asarray(var))
    
        # Stack variables into a single array if multiple fields are used
        if self.ndim > 1:
            # Allocate output array: [time, variable, lat, lon]
            nt, ny, nx = vars_np[0].shape
            data_comp = np.zeros((nt, self.ndim, ny, nx), dtype=vars_np[0].dtype)
    
            # Fill variable dimension
            for i in range(self.ndim):
                data_comp[:, i, :, :] = vars_np[i]
    
        else:
            # Single variable case: return [time, lat, lon]
            data_comp = vars_np[0]

        return data_comp


    def remove_doublon(self):
        "Removes duplicate timestamps and associated data"

        # Initialize mask (1 = keep, 0 = remove duplicate)
        mask_doublon = np.ones(self.time.shape[0])
        
        # Identify consecutive duplicate timestamps
        for i in range(1, self.time.shape[0]):
            if self.time[i] == self.time[i-1]:
                mask_doublon[i] = 0
                print(self.time[i])
        
        # Convert mask to boolean
        mask_doublon = mask_doublon == 1
        
        # Apply mask consistently to all time-dependent variables
        self.pr_ts = self.pr_ts[mask_doublon]
        self.time = self.time[mask_doublon]
        self.data = self.data[mask_doublon, :, :]
        self.datap = self.datap[mask_doublon, :, :]


    def clip_data(self):
        "Clips spatial fields to the domain used for z500 histogram computations"

        # Display initial shapes for debugging
        print(self.lons.shape, self.mask_sp.shape)
        
        # Apply spatial mask to longitude, latitude, and pixel area
        self.lons = ct.clipping_data(self.lons, self.mask_sp)
        self.lats = ct.clipping_data(self.lats, self.mask_sp)
        self.pix_area = ct.clipping_data(self.pix_area, self.mask_sp)
        
        print('Clipped domain shape:', self.lons.shape)
        
        # Note: data clipping is optional and currently disabled
        # self.data = ct.clipping_data(self.data, self.mask_sp)

    
    def fig_col(self, date_looked=datetime(2010, 10, 10),
                geop2plot=np.zeros(0),
                mask_cut_off_low=np.zeros(0),
                spatial_delim_mask=np.zeros(0),
                lons_gf=np.zeros(0),
                lats_gf=np.zeros(0),
                size_scat_gf=6,
                dates_gf=np.array([]),
                mask_time_COL=np.zeros(0),
                xs=np.zeros(0),
                ys=np.zeros(0),
                step=2,
                vdown=0,
                vup1=0,
                vup=0, th_precip=0,
                vexm=5660):
        """
        Generates a diagnostic figure illustrating the algorithm behavior
        and the detection of cut-off lows for a given date
        """
    
        # ============================================================
        # FIGURE INITIALIZATION
        # ============================================================
        fig = plt.figure(figsize=(20, 10))
    
        # Main map (Robinson projection)
        ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.Robinson())
    
        # Display current date on map
        ax1.text(.05, .03, date_looked.strftime("%Y-%m-%d"), fontsize=18, weight='bold',
                 transform=ax1.transAxes, bbox=dict(color='white', alpha=.3))
    
        cmap = cm.jet  # colormap for geopotential
    
    
        # ============================================================
        # COLOR SCALE FOR Z500
        # ============================================================
        bm = np.nanmin(geop2plot)
        bma = np.nanmax(geop2plot)
    
        step1 = 10  # contour interval
    
        # Define bounds rounded to step
        bound_min = bm-bm % step1
        bound_up = bma+(step1-(bma % step1))
    
        bounds = np.arange(bound_min, bound_up+step1, step1)
    
        mfont = 10  # font size for annotations
    
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
        # Plot geopotential field
        pl = ax1.pcolormesh(self.lons_nomodif, self.lats_nomodif, geop2plot,
                            transform=ccrs.PlateCarree(), norm=norm, cmap=cmap)
    
        # Overlay detected regions (makak labels, invisible here)
        ax1.pcolormesh(self.lons, self.lats, self.makak,
                       transform=ccrs.PlateCarree(), alpha=.0)
    
    
        # ============================================================
        # CONTOURS (ALGORITHM DIAGNOSTICS)
        # ============================================================
    
        # Mask of detected closed regions
        CS = ax1.contour(self.lons, self.lats, mask_cut_off_low,
                          transform=ccrs.PlateCarree(), cmap=cm.cool, linewidths=5, zorder=2)
    
        # Upper bound contour
        CS = ax1.contour(self.lons_nomodif, self.lats_nomodif, geop2plot < vup1,
                         transform=ccrs.PlateCarree(), cmap=cm.Greys_r,
                         linewidths=5, linestyles='--')
    
        # Lower bound contour
        CS = ax1.contour(self.lons_nomodif, self.lats_nomodif, geop2plot < vdown,
                         transform=ccrs.PlateCarree(), cmap=cm.Greys_r,
                         linewidths=5, linestyles='dashdot')
    
    
        # ============================================================
        # COLORBAR + HISTOGRAM AXES
        # ============================================================
        wax=.237
        xax=.182
    
        # Colorbar axis
        c_ax = fig.add_axes([xax, .37, wax, .03])
    
        # Histogram axis
        ax = fig.add_axes([xax, .4, wax, .1])
    
        cbar = plt.colorbar(pl, cax=c_ax, shrink=.6, orientation='horizontal')
        cbar.set_label('z500 [m]')
    
    
        # ============================================================
        # MAP FEATURES
        # ============================================================
        ax1.coastlines(alpha=1, zorder=3)
        ax1.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=1)
    
        # Plot detected COL centers
        ax1.scatter(lons_gf, lats_gf, s=size_scat_gf,
                    marker='*', transform=ccrs.PlateCarree(),
                    zorder=10, color='black')
    
        # Number each detected COL
        n = np.arange(1, np.size(lons_gf)+1)
        for i, txt in enumerate(n):
            ax1.annotate(txt, (lons_gf[i], lats_gf[i]),
                          transform=ccrs.PlateCarree(),
                          fontsize=mfont, weight='bold',
                          bbox=dict(color='white', alpha=.2))
    
    
        # ============================================================
        # TIME AXIS (UNUSED / LEGACY)
        # ============================================================
        xss = xss = np.array([datetime(2012, 1, 1),
                              datetime(2012, 2, 1),
                              datetime(2012, 3, 1),
                              datetime(2012, 4, 1),
                              datetime(2012, 5, 1),
                              datetime(2012, 6, 1),
                              datetime(2012, 7, 1),
                              datetime(2012, 8, 1),
                              datetime(2012, 9, 1),
                              datetime(2012, 10, 1),
                              datetime(2012, 11, 1),
                              datetime(2012, 12, 1)])
    
        # Replaced by simple index (not used later)
        xss = np.arange(12)
    
        lw = 2  # line width
    
    
        # ============================================================
        # HISTOGRAM OF Z500
        # ============================================================
        ax.bar(self.b, self.a, zorder=1, width=step, alpha=.5, color='black')
    
        ax.set_xticks([])  # hide x ticks
        ax.set_title('z500 histogram', fontsize=16, loc='left')
    
        ax.set_ylim([0, np.max(self.a)+np.max(self.a)/10])
        ax.set_xlim([bound_min, bound_up])
    
    
        # ============================================================
        # THRESHOLDS + CURVES
        # ============================================================
        ax2 = 1  # placeholder (kept as-is)
    
        lss = ['dashdot', '--','-']  # line styles
        nms = ['Lower threshold', 'Upper threshold','Maximal extension']
        colors=['black','black','indigo']
    
        suivp = -1
    
        # Plot lower and upper thresholds
        for x in np.array([vdown, vup1]):
            suivp += 1
    
            ax.plot(np.ones_like(np.arange(0, 100, .001))*x,
                    np.arange(0, 100, .001),
                    color=colors[suivp],
                    linestyle=lss[suivp],
                    zorder=2, lw=lw,
                    label=nms[suivp])
    
    
        # Plot histogram curve (extrema detection)
        ax.plot(xs[np.isfinite(ys)], ys[np.isfinite(ys)],
                lw=lw, color='black', zorder=5)
    
        # Plot selected isohypse
        ax.plot(np.ones_like(np.arange(0, 100, .001))*vup,
                np.arange(0, 100, .001),
                color='magenta', linestyle='-',
                zorder=4, lw=lw,
                label='Drawn isohypse')
    
        # Legend positioning (outside axis)
        ax.legend(loc=[-.445, 3],frameon=False)
    
    
        # ============================================================
        # RETURN OBJECTS
        # ============================================================
        ax2=1  
    
        return(fig, ax, ax1, ax2)


    def research_bounds2(self, freqs, centers, map2w=np.zeros(0)):
        """
        Identifies lower and upper bounds of z500 values associated with COL occurrence
        by detecting inflection points in an interpolated frequency distribution.
        """

        # Perform cubic spline interpolation of the histogram curve
        cs = CubicSpline(freqs, centers)
        
        # Define a high-resolution x-axis to smooth and refine the curve
        xs = np.arange(5000, 6500, 0.5)
        ys = cs(xs)   # Interpolated (smoothed) curve

        # Define valid bounds based on the input field (if provided)
        # Otherwise, use default z500 range
        minv, maxv = np.nanmin(map2w), np.nanmax(map2w) if map2w.size else (5000, 6500)

        # Compute first derivatives using finite differences
        dy_prev = ys[1:-1] - ys[:-2]
        dy_next = ys[2:] - ys[1:-1]

        # Detect inflection points where derivative changes sign
        # Also require sufficient amplitude (ys > 0.8) to filter noise
        inflection = (dy_prev * dy_next < 0) & (ys[1:-1] > 0.8)
        x_candidates = xs[1:-1][inflection]

        # Keep only inflection points within the valid z500 range
        x_valid = x_candidates[(x_candidates >= minv) & (x_candidates <= maxv)]

        # Select bounds depending on the number of detected inflection points

        if x_valid.size == 0:
            # No valid inflection → return NaNs and default values
            return float('nan'), 0, xs, ys, float('nan')

        elif x_valid.size == 1:
            # Only one inflection → use domain minimum as lower bound
            return minv, x_valid[0], xs, ys, x_valid[0]

        elif x_valid.size == 2:
            # Two inflections → directly define lower and upper bounds
            return x_valid[0], x_valid[1], xs, ys, x_valid[1]

        elif x_valid.size == 3:
            # Three inflections → discard first (likely noise), keep next two
            return x_valid[1], x_valid[2], xs, ys, x_valid[-1]

        elif x_valid.size == 4:
            # Four inflections → select central pair as main bounds
            return x_valid[1], x_valid[2], xs, ys, x_valid[-1]

        else:
            # More than four inflections → use heuristic selection
            vdd = x_valid[1]          # lower bound
            vupp1 = x_valid[2]       # upper bound
            vmax_ext = x_valid[4]    # extended upper limit

        return vdd, vupp1, xs, ys, vmax_ext
    
            
    def col(self, nneofs=5, th=30, dates=np.array(datetime(1982, 10, 7)),
                nneigh=0, ste=10, msm=20, mfont=10,
                critere_taille=20, lon1=-20, lon2=28, lat1=35, lat2=60,
                thp=1,
                ff='/srv5_tmp2/jbrajkovic/forgif/',
                plt_or_not=1,
                Dcrit=1500,criteria=None,
                showpl=0):
            
            """
            Detect and attribute cut-off lows (COLs) from geopotential height (z500)
            fields and associate them with extreme precipitation events.
            
            This function implements a contour-based detection algorithm using
            histogram-derived thresholds to identify closed low-pressure systems
            (cut-off lows, COLs) in z500 fields. It then attributes precipitation
            events to the nearest detected COL based on spatial proximity and
            geometrical constraints.
            
            The method operates independently for each date and scans multiple
            isohypses to detect enclosed structures. Both precipitation-attributed
            COLs and all detected COLs (including dry ones) are stored.
            
            Parameters
            ----------
            nneofs : int, optional
                Reserved parameter (currently unused).
            
            th : float, optional
                Reserved parameter (currently unused).
            
            dates : array-like of datetime
                Dates over which the COL detection algorithm is applied.
            
            nneigh : int, optional
                Reserved parameter (currently unused).
            
            ste : float
                Step size (in geopotential meters) used to scan isohypses between
                detected lower and upper bounds.
            
            msm : float
                Marker size used in diagnostic plots.
            
            mfont : float
                Font size used in diagnostic plots.
            
            critere_taille : float
                Minimum area threshold (in km²) required for a closed contour
                to be considered a valid COL.
            
            lon1, lon2 : float
                Longitude bounds defining the spatial domain for valid COL centers.
            
            lat1, lat2 : float
                Latitude bounds defining the spatial domain for valid COL centers.
            
            thp : float
                Precipitation threshold used to pre-select candidate extreme events.
            
            ff : str
                Output directory where diagnostic figures are saved.
            
            plt_or_not : int (0 or 1)
                If 1, diagnostic plots of the detection process are generated
                and saved.
            
            showpl : int (0 or 1)
                If 1, plots are displayed interactively.
            
            Dcrit : float
                Characteristic distance scale (in km) used in the attribution
                criteria between COL centers and precipitation.
            
            criteria : float, optional
                Percentile used to define the precipitation threshold.
                If None, the default percentile `self.qu` is used.
            
            Returns
            -------
            vec_years : ndarray
                Array of years covered by the dataset.
            
            n_event_years : ndarray
                Number of COL-attributed precipitation events per year.
            
            compil_lons : ndarray
                Longitudes of COL centers associated with precipitation events.
            
            compil_lats : ndarray
                Latitudes of COL centers associated with precipitation events.
            
            compil_dates1 : ndarray
                Dates of COL-attributed precipitation events.
            
            vups : ndarray
                Isohypse values at which COLs were detected.
            
            compil_present_COLS_all_lons : ndarray
                Longitudes of all detected COLs (including non-precipitating ones).
            
            compil_present_COLS_all_lats : ndarray
                Latitudes of all detected COLs (including non-precipitating ones).
            
            all_col_even_dry_ones_dates_all : ndarray
                Dates of all detected COLs, including dry events.
            
            Notes
            -----
            - Detection is based on identifying closed contours in z500 fields
              using histogram-derived thresholds.
            - The algorithm scans multiple isohypses to identify enclosed regions.
            - Attribution relies on spatial proximity between COL centers and
              precipitation centroids.
            - Results are sensitive to parameter choices (e.g., `ste`,
              `critere_taille`, `criteria`, `Dcrit`).
            - Diagnostic plotting can significantly slow down execution when enabled.
            - The function preserves all detected COLs, including those not
              associated with precipitation (dry COLs).
                        """
            if self.interpolate_z500==1:
                # Save original latitude/longitude grid (before interpolation)
                lats_old=self.lats
                lons_old=self.lons
            
                # Replace current grid with ERA reference grid
                self.lons=np.copy(self.lons_ERA)
                self.lats=np.copy(self.lats_ERA)
            
                # Keep a copy of the "non-modified" grid (for plotting or reference)
                self.lons_nomodif=np.copy(self.lons)
                self.lats_nomodif=np.copy(self.lats)
            
                # Define new grid explicitly (used later for interpolation)
                lons_new=np.copy(self.lons_ERA)
                lats_new=np.copy(self.lats_ERA)
            
                # Spatial mask defining the region of interest for analysis
                self.mask_sp = (self.lons > self.lon_etendue1) & (self.lons < self.lon_etendue2) & (
                    self.lats > self.lat_etendue1) & (self.lats < self.lat_etendue2)
                
                # Compute latitudinal resolution of the grid
                self.res_lat = abs(self.lats[0, 0]-self.lats[1, 0])
            
                # Initialize pixel area array (same shape as grid)
                self.pix_area= np.ones_like(self.lons)
            
                # If resolution was zero (edge case), compute it along other dimension
                if self.res_lat == 0:
                    self.res_lat = abs(self.lats[0, 0]-self.lats[0, 1])
            
                # Compute approximate pixel area (km²)
                # Uses spherical approximation: 1° ≈ 111 km and cosine(latitude)
                self.pix_area[:, :] = (
                    self.res_lat*np.cos(np.deg2rad(self.lats)))*111*(self.res_lat*111)
        
            
            "**************************************************"
            "** Clipping for not having pixels too far north **"
            "**************************************************"
            
            # Apply spatial clipping (removes unwanted regions, e.g. polar areas)
            self.clip_data()
            
            # Print shape of data after clipping (diagnostic)
            print(f'Data shape after clipping: {self.lons.shape}')
            # print(f'rap_taille and self.lats.shape : {self.lats.shape[0]} ')
            
            # Build array of years covered by dataset
          
            vec_years = np.arange(np.min(self.time.year), np.max(self.time.year)+1)
            
            # Initialize counter: number of detected events per year
            n_event_years = np.zeros_like(vec_years)
            
            # Adjust scaling parameter if needed (specific to algorithm tuning)
            if self.rap_taille-1 > .1:
                self.rap_taille = (self.rap_taille+1)*2
            
            "****************************"
            "** Initialising variables **"
            "****************************"
            
            # Local copy of pixel area
            pix_area = self.pix_area
            
            # Boolean-like array to mark precipitation events above threshold
            got_fris = np.zeros_like(self.pr_ts[self.pr_ts > thp])
            
            # Counter for saved figures
            suiv = 100
            
            # Unused / placeholder variable
            ch = 0
            
            # Arrays storing COLs associated with precipitation
            compil_lons = np.zeros(0)
            compil_lats = np.zeros(0)
            compil_dates = np.zeros(0)
            
            # Re-initialize counters (redundant but kept as-is)
            suiv = 100
            ch = 0
            
            # Additional storage arrays
            compil_dates1 = np.zeros(0)  # dates of attributed events
            vups = np.zeros(0)           # isohypse levels at detection
            
            # Storage for ALL detected COLs (including dry ones)
            compil_present_COLS_all_lons=np.array([])
            compil_present_COLS_all_lats=np.array([])
            all_col_even_dry_ones_dates_all=np.array([])
            
            # Initialize previous year tracker (used for printing/logging)
            cur_y1=(dates[0].year)-1
            
            # ==========================================================
            # LOOP OVER ALL DATES
            # ==========================================================
            for date2p in dates:
            
                # Print current date being processed
                print(date2p)
            
                # Flag: whether a precipitation event has already been attributed for this date
                already_found=0
            
                # Current year
                cur_y=date2p.year
            
                # If new year → print summary of detected COLs so far
                if cur_y!=cur_y1:
                    cur_y1=cur_y
                    print(cur_y1,'identified COLS',compil_present_COLS_all_lats.shape)
            
                # Print month at the start of each month
                if date2p.day==1:
                    print(f'month: {date2p.month}')
                    
                # ------------------------------------------------------
                # Storage for COLs detected on THIS date (including dry)
                # ------------------------------------------------------
                compil_present_COLs=np.array([])
                compil_present_COLs_lons=np.array([])
                compil_present_COLs_lats=np.array([])
                all_col_even_dry_ones_dates=np.array([])
                
              
                # Convert Python datetime → numpy datetime64 (for xarray selection)
                date2p64 = np.datetime64(date2p)
                date2p64 = np.datetime64(date2p)  # duplicated line (kept as-is)
            
            
                # ------------------------------------------------------
                # LOAD Z500 FIELD
                # ------------------------------------------------------
                v2pl = self.ds_data['z500'].sel(time=date2p64, method='nearest').values
            
                # If values are too large → likely geopotential → convert to height
                # print(np.nanmax(v2pl))
                if np.nanmax(v2pl)>10000:
                    v2pl/=9.81
            
                # Optional interpolation to ERA grid
                if self.interpolate_z500==1:
                    print('interpolation')
                    v2pl=(ct.interpolation_grid_2d(v2pl.T,lats_new_grid=lats_new.T,lons_new_grid=lons_new.T,
                                                                  lats_old_grid=lats_old.T,lons_old_grid=lons_old.T)).T
            
                # Keep a copy of original field (used later for plotting)
                v2pl_nomodif=np.copy(v2pl)
            
                # Apply spatial clipping mask
                v2pl = ct.clipping_data(v2pl, self.mask_sp)
                
                
                # ------------------------------------------------------
                # COMPUTE QUANTILES OF Z500 (used later for histogram)
                # ------------------------------------------------------
                a = np.quantile(v2pl, np.arange(0, 1, .01))
                
            
                # ------------------------------------------------------
                # LOAD PRECIPITATION FIELD
                # ------------------------------------------------------
                pr2pl=self.dsp['Precip'].sel(time=date2p64, method='nearest').values
            
                # Compute precipitation threshold (percentile-based)
                if criteria is None:
                    criteria1=np.nanquantile(pr2pl[self.maskp],self.qu/100)
                else:
                    criteria1=np.nanquantile(pr2pl[self.maskp],criteria/100)
                
                "Mean position of the precipitation front"
                
                # Extract pixels exceeding precipitation threshold
                lon_pr1 = self.lons1[(pr2pl > criteria1) & (self.maskp)]
                lat_pr1 =  self.lats1[(pr2pl > criteria1) & (self.maskp)]
            
                # Compute centroid of precipitation region
                lonpr_test=np.mean(lon_pr1)
                latpr_test=np.mean(lat_pr1)
                
                "Zones where enclosed zones are being searched after"
                
                # Define spatial domain where COL detection is allowed
                mas1 = (self.lons > lon1) & (self.lons < lon2)\
                        & (self.lats < lat2) & (self.lats > lat1)
                        
            
                # Create output directory if it does not exist
                if os.path.exists(ff) == False:
                    os.mkdir(ff)
            
                # Line width for plotting
                lw = 2
            
                # Save initial figure counter
                start_suiv = suiv
              
                "already checked values are stored to increase the algo speed"
            
                # Store already processed isohypses to avoid redundant computations
                val_al_checked = np.zeros(0)
                "**********************************************************************"
                "** Looping through different class sizes for constructing histogram **"
                "**********************************************************************"
                
                # Reset figure counter for this new histogram loop
                suiv = start_suiv
                
                # Loop over different histogram bin sizes (step controls smoothing / detection sensitivity)
                for st in np.arange(20, 150, 10):
                    
                    
                    # Build histogram of z500 values between 5000 and 6500 with bin size = st
                    # self.a → histogram counts, self.b → bin edges
                    self.a, self.b = ct.make_hist(
                        v2pl, vmin=5000, vmax=6500, step=st)
                
                    # Convert histogram counts to percentage (or scaling for detection)
                    self.a = self.a*100
                
                    "Recherches des Extremas"
                
                    # Copy field (avoid modifying original during processing)
                    v2pl1 = np.copy(v2pl)
                
                    # Detect lower and upper bounds of potential COL structures
                    # vd   → lower bound
                    # vup1 → upper bound
                    # xs, ys → coordinates of extrema
                    # v_extmax → maximum extremum value
                    vd, vup1, xs, ys, v_extmax = self.research_bounds2(
                        self.b, self.a, map2w=v2pl1)
                    
                    
                    # If no valid lower bound found → skip this histogram configuration
                    if np.isfinite(vd) == False:
                        continue
                
                    # print(st)   
                
                    "Removing edges for finding enclosed zones"
                
                    # Mask to remove domain edges (prevents detecting open contours touching boundaries)
                    mas2 = (self.lons > np.nanmin(self.lons)*1.01) & (self.lons < np.nanmax(self.lons)*.98)\
                        & (self.lats < np.nanmax(self.lats)*.98) & (self.lats > np.nanmin(self.lats)*1.01)
                        
                    "Will become 1 if event attributed to a cut-off low"
                
                    # Flag indicating whether a COL is successfully linked to precipitation
                    gotta_fria = 0
                    
                    # Loop over isohypses between vd and vup1 with step "ste"
                    for vup in np.arange(vd, vup1, ste):
                        # print(vup)
                        
                        "Selecting lats ans lons constituting the curren isohypse"
                
                        # Extract coordinates of points close to current isohypse level
                        lonext=self.lons[(v2pl1>vup-1)&(v2pl1<vup+1)]
                        latext=self.lats[(v2pl1>vup-1)&(v2pl1<vup+1)]
                        
                
                        # Copy field for manipulation
                        premval = np.copy(v2pl1)
                
                        # Identify if this isohypse (or a similar one) has already been processed
                        maskval = (val_al_checked > vup-ste /
                                   2) & (val_al_checked < vup+ste/2)
                        
                        "Checking whether the current isohypse has already been watched"
                
                        # Skip if already processed (optimization to avoid redundant work)
                        if np.size(val_al_checked[maskval] > 0):
                            continue
                        else:
                            # Store this isohypse as processed
                            val_al_checked = np.append(val_al_checked, vup)
                        
                        "Looking for enclosed zones with labels (1,2,...n zone(s))"
                
                        # Detect closed contours (candidate COL regions)
                        # self.makak contains labeled regions (1,2,...)
                        self.makak = ct.delimitate_closed_zones_optimized(
                            premval, vmax=vup, mas1=mas2)
                        
                        "Putting Nans where required"
                
                        # Mask values below the current isohypse (focus on enclosed structures)
                        premval[(premval > 0) & (premval < vup)] = float('nan')
                
                        # Compute area of region labeled "1" (initial reference region)
                        gf_area = np.sum(pix_area[self.makak == 1])
                
                        # Build binary mask of detected structure (1 where masked / inside region)
                        mas = np.zeros_like(v2pl)
                        mas[np.isfinite(premval) == False] = 1
                        
                        
                        # Generate a plot of the algo evolution if required
                        if plt_or_not:
                            print('I am plotting',vup)
                        
                            # Create figure with multiple panels (z500, masks, diagnostics)
                            fig, ax, ax1, ax2 = self.fig_col(date_looked=date2p,
                                                             geop2plot=v2pl_nomodif,
                                                             mask_cut_off_low=mas,
                                                             spatial_delim_mask=mas1,
                                                             lons_gf=compil_lons,
                                                             lats_gf=compil_lats,
                                                             size_scat_gf=msm,
                                                             dates_gf=compil_dates,
                                                             mask_time_COL=got_fris,
                                                             xs=xs,
                                                             ys=ys,
                                                             step=st,
                                                             vdown=vd,
                                                             vup1=vup1,
                                                             vup=vup,
                                                             th_precip=thp,
                                                             vexm=v_extmax)
                        
                            # Display whether an event has already been attributed
                            if already_found:
                                EV_att='YES'
                                tcol='green'
                            else:
                                EV_att='NO'
                                tcol='darkred'
                        
                            # Add annotation box with number of detected COLs and attribution status
                            ax1.text(.44,.01,
                                     f'Identified COLs: {all_col_even_dry_ones_dates.shape[0]}\nEvent attribution:',
                                     fontsize=14,weight='bold',
                                     bbox=dict(color='white',alpha=.5),
                                     transform=ax1.transAxes)
                        
                            # Add YES/NO label
                            ax1.text(.88,.01,f'{EV_att}',
                                     fontsize=14,weight='bold',color=tcol,
                                     bbox=dict(color='white',alpha=.5),
                                     transform=ax1.transAxes)
                        
                            # Draw spatial bounding box of study region
                            ax1.plot(np.arange(lon1, lon2+.01, .01),
                                     np.ones_like(np.arange(lon1, lon2+.01, .01))*lat2,
                                     transform=ccrs.PlateCarree(), lw=5, color='grey', linestyle='--')
                        
                            ax1.plot(np.arange(lon1, lon2+.01, .01),
                                     np.ones_like(np.arange(lon1, lon2+.01, .01))*lat1,
                                     transform=ccrs.PlateCarree(), lw=5, color='grey', linestyle='--')
                        
                            ax1.plot(np.ones_like(np.arange(lat1, lat2+.01, .01))*lon1,
                                     np.arange(lat1, lat2+.01, .01),
                                     transform=ccrs.PlateCarree(), lw=5, color='grey', linestyle='--')
                        
                            ax1.plot(np.ones_like(np.arange(lat1, lat2+.01, .01))*lon2,
                                     np.arange(lat1, lat2+.01, .01),
                                     transform=ccrs.PlateCarree(), lw=5, color='grey', linestyle='--',
                                     label='Research perimeter')
                                                                            
                        "Which enclosed zone is the closest to the precip front?"
                        
                        # Get all unique labels of detected enclosed zones (ignore background = 0)
                        n_enclosed_zones=np.unique(self.makak) 
                        n_enclosed_zones=n_enclosed_zones[n_enclosed_zones>0]
                        
                        
                        # Initialize distance trackers
                        min_dis2precip=-10E6   # (unused here but kept as-is)
                        dis2rain=10E9          # large initial value → will store minimum distance
                        
                        # Initialize variables for selected COL center
                        lat_gf=0
                        lon_gf=0
                        
                        # Loop over all detected enclosed zones
                        for icol in n_enclosed_zones:
                            
                            # Extract lat/lon points belonging to this enclosed zone
                            lats_gf1 = self.lats[self.makak == icol]
                            lons_gf1 = self.lons[self.makak == icol]
                        
                        
                            # Compute centroid (mean position) of the zone
                            lat_gf1=np.mean(lats_gf1)
                            lon_gf1=np.mean(lons_gf1)
                        
                            # Compute distance (in km) between zone centroid and precipitation centroid
                            dis2rain1 = ct.dis2pix(lat_gf1, lon_gf1, latpr_test, lonpr_test)/1000
                            
                            # Keep the zone closest to precipitation
                            if dis2rain1<dis2rain:
                                dis2rain=dis2rain1
                        
                                # Store coordinates of closest zone
                                lat_gf = lat_gf1
                                lon_gf = lon_gf1
                        
                                # Compute its area
                                gf_area = np.sum(pix_area[self.makak == icol])
                        
                                # Store full spatial extent of that zone
                                lats_gf=np.copy(lats_gf1)
                                lons_gf=np.copy(lons_gf1)
                        
                        "Adding features to the figure if required"
                                
                        # ----------------------------------------------------------
                        # PLOTTING: add precipitation and labels on diagnostic figure
                        # ----------------------------------------------------------
                        if plt_or_not:
                        
                            # Plot precipitation pixels exceeding threshold
                            ax1.scatter(lon_pr1, lat_pr1, marker='x', s=5,
                                        transform=ccrs.PlateCarree(), color='lightgrey',
                                        label=f'precipitation pixels\n> percentile {criteria}'+' ($q_{rain}$)'+f'\n= {criteria1:.1f}mm',
                                        zorder=5)
                        
                            # Add subplot labels (panel identifiers)
                            ax1.text(-.1, 1, 'a)', fontsize=12,
                                     weight='bold', transform=ax1.transAxes)
                        
                            ax.text(-.1, 1, 'b)', fontsize=12,
                                    weight='bold', transform=ax.transAxes)
                                                    

                        
                        # Check if the selected enclosed zone exceeds the minimum area threshold
                        if gf_area > critere_taille:
                            
                            "Adding to all the stored enclosed zones if new ones"
                        
                            # Get all detected enclosed zones (labels > 0)
                            n_cols=np.unique(self.makak)
                            n_cols=n_cols[n_cols>0]
                            
                            # print(n_cols)
                        
                            # Loop over all detected zones to store them (including dry COLs)
                            for indice_col in n_cols:
                        
                                # Check if the current zone satisfies the size criterion
                                if np.sum(pix_area[self.makak == indice_col])>critere_taille:
                        
                                    # Compute centroid of the COL
                                    lon_col=np.nanmean(self.lons[self.makak==indice_col])
                                    lat_col=np.nanmean(self.lats[self.makak==indice_col])
                                    
                                    # Keep only COLs within predefined spatial domain
                                    if lon_col >=lon1 and lon_col<= lon2 and lat_col>=lat1 and lat_col<=lat2:
                        
                                        # If this is the first COL detected for this date → add directly
                                        if compil_present_COLs_lons.shape[0]==0:
                                            compil_present_COLs_lons=np.append(compil_present_COLs_lons,lon_col)
                                            compil_present_COLs_lats=np.append(compil_present_COLs_lats,lat_col)
                                            all_col_even_dry_ones_dates=np.append(all_col_even_dry_ones_dates,date2p)
                        
                                        else:
                                            # print('shapes 2 watch',compil_present_COLS_all_lats.shape,compil_present_COLS_all_lons.shape,lat_col,lon_col)
                        
                                            # Compute distance to already stored COLs (avoid duplicates)
                                            dists2other_cols=ct.dis2pix(
                                                lat_col, lon_col,
                                                compil_present_COLs_lats,
                                                compil_present_COLs_lons)/1000
                        
                                            # print('inf dists',dists2other_cols[dists2other_cols<500])
                        
                                            # Only add if no existing COL is closer than 500 km
                                            if dists2other_cols[dists2other_cols<500].shape[0]==0:
                                                print('appending')
                                                compil_present_COLs_lons=np.append(compil_present_COLs_lons,lon_col)
                                                compil_present_COLs_lats=np.append(compil_present_COLs_lats,lat_col)
                                                all_col_even_dry_ones_dates=np.append(all_col_even_dry_ones_dates,date2p)
                        
                                else:
                                    # Skip zones that are too small
                                    continue
                                
                        
                            
                            
                            "distance between closest precip point and COL center in kilometers"
                        
                            # Compute distance between ALL precip pixels and COL center
                            disex2precip=ct.dis2pix(lat_pr1, lon_pr1, lat_gf, lon_gf)/1000
                        
                            # Identify the precipitation pixel closest to the COL center
                            lat_pr,lon_pr=lat_pr1[disex2precip==np.nanmin(disex2precip)][0],\
                                           lon_pr1[disex2precip==np.nanmin(disex2precip)][0]
                         
                            # Distance between COL center and closest precipitation pixel
                            dis2rain = ct.dis2pix(lat_gf, lon_gf, lat_pr, lon_pr)/1000
                            
                            
                            # Distance from closest precip point to isohypse contour points
                            disex2precip=ct.dis2pix(lat_pr, lon_pr, latext, lonext)
                        
                            # Distance from COL center to isohypse contour points
                            disex2precip1=ct.dis2pix(lat_gf, lon_gf, latext, lonext)
                        
                            # Combined metric (used to find relevant contour point)
                            disex2precip=disex2precip1+disex2precip
                            
                            
                            # Find contour point minimizing combined distance
                            try:
                                lata,lona=latext[disex2precip==np.nanmin(disex2precip)][0],\
                                          lonext[disex2precip==np.nanmin(disex2precip)][0]
                            except:
                                # Catch potential indexing errors (e.g., empty arrays)
                                print(' error disex2precip')
                                
                            
                           
                            # Distance between COL center and selected isohypse point (in km)
                            disex2gf=ct.dis2pix(lata, lona, lat_gf ,lon_gf)/1000
                                                    
                                                    
                            if plt_or_not:
                                # ----------------------------------------------------------
                                # PLOTTING: visualize COL center and closest precipitation
                                # ----------------------------------------------------------
                            
                                # Plot COL center (black cross, larger size for emphasis)
                                ax1.scatter(lon_gf, lat_gf, marker='x', s=200,
                                            transform=ccrs.PlateCarree(), color='black',
                                            linewidths=3, zorder=5)
                            
                                # Plot COL center again (magenta cross, labeled)
                                ax1.scatter(lon_gf, lat_gf, marker='x', s=100,
                                            transform=ccrs.PlateCarree(), color='magenta',
                                            linewidths=3,
                                            label='COL center', zorder=5)
                            
                                # Plot closest precipitation pixel (black triangle, slightly shifted for visibility)
                                ax1.scatter(lon_pr-.1, lat_pr+.2,
                                            transform=ccrs.PlateCarree(), s=220,
                                            marker='^', color='black', zorder=5)
                            
                                # Plot closest precipitation pixel again (magenta triangle, labeled)
                                ax1.scatter(lon_pr, lat_pr,
                                            transform=ccrs.PlateCarree(), s=100,
                                            marker='^',
                                            label='Closest precipitation\npixel',
                                            color='magenta', zorder=5)
                            
                            
                            # ----------------------------------------------------------
                            # FINAL ATTRIBUTION CONDITION (COL ↔ precipitation)
                            # ----------------------------------------------------------
                            # Conditions:
                            # 1. COL center must be inside spatial domain
                            # 2. Distance COL ↔ precipitation must be smaller than
                            #    COL ↔ isohypse distance (geometric consistency)
                            #    (optional stricter condition with Dcrit is commented out)
                            if lat_gf < lat2 and lat_gf > lat1 and lon_gf < lon2 and lon_gf > lon1 and dis2rain<1.*disex2gf:#and dis2rain < Dcrit:
                                gotta_fria = 1

                            if gotta_fria == 1 and already_found==0:
                                # ----------------------------------------------------------
                                # STORE ATTRIBUTED COL (only once per date)
                                # ----------------------------------------------------------
                            
                                # Mark that an event has already been attributed for this date
                                already_found=1
                            
                                # Store COL center coordinates
                                compil_lats = np.append(compil_lats, lat_gf)
                                compil_lons = np.append(compil_lons, lon_gf)
                            
                                # Store isohypse level at which COL was detected
                                vups = np.append(vups, vup)
                            
                                # (Alternative storage commented out by author)
                                # compil_lons=np.append(compil_lons,np.mean(lons_gf))
                            
                                # Mark this date as having a precipitation event linked to a COL
                                got_fris[self.time[self.pr_ts > thp] == date2p] = 1
                            
                                # Store event date
                                compil_dates1 = np.append(compil_dates1, date2p)
                              
                                # Multiply mask with precipitation time series (keeps only detected events)
                                self.gf = got_fris*self.pr_ts[self.pr_ts > thp]
                            
                                # Increment yearly counter of attributed events
                                n_event_years[vec_years == date2p.year] += 1
                            
                                # ----------------------------------------------------------
                                # OPTIONAL: store date for plotting (fixed year for alignment)
                                # ----------------------------------------------------------
                                if plt_or_not:
                            
                                    # Store date with fixed year (2012) for visualization consistency
                                    compil_dates = np.append(compil_dates, datetime(
                                        2012, date2p.month, date2p.day))
                            
        
                        # ----------------------------------------------------------
                        # SAVE FIGURE IF PLOTTING ENABLED
                        # ----------------------------------------------------------
                        if plt_or_not:
                            print(f'saving to {ff}')
                        
                            # Add legend outside plot
                            ax1.legend(loc=[-.46, .05],frameon=False)
                        
                            # Adjust subplot spacing
                            plt.subplots_adjust(hspace=.25)
                        
                            # Save figure with incremental filename
                            plt.savefig(ff+'fig_'+str(int(suiv)) +
                                        '.png', bbox_inches='tight')
                        
                            # Increment figure counter
                            suiv+=1
                        
                            # Display figure
                            if showpl:
                                plt.show()
                        
                        
                # ----------------------------------------------------------
                # STORE ALL DETECTED COLS (INCLUDING DRY ONES)
                # ----------------------------------------------------------
                all_col_even_dry_ones_dates_all=np.append(
                    all_col_even_dry_ones_dates_all,
                    all_col_even_dry_ones_dates)
                
                compil_present_COLS_all_lons=np.append(
                    compil_present_COLS_all_lons,
                    compil_present_COLs_lons)
                
                compil_present_COLS_all_lats=np.append(
                    compil_present_COLS_all_lats,
                    compil_present_COLs_lats)
                    
                            
            # ----------------------------------------------------------
            # RETURN FINAL OUTPUTS
            # ----------------------------------------------------------
            return(vec_years, n_event_years, compil_lons, compil_lats, compil_dates1,
               vups,compil_present_COLS_all_lons,
               compil_present_COLS_all_lats,all_col_even_dry_ones_dates_all)
           
     

    def COL_research(self, plt_or_not=0, mod='ERA5', ts=72, thp=1, thf=1, dirout='/srv5_tmp2/jbrajkovic/',
                     critere_taille=300000, ste=20, numero='',
                     lat_max=55,
                     lat_min=35,
                     lon_max=40,
                     lon_min=-10,
                     test_mode=0,
                     d2p=None, criteria=None,
                     showpl=0):
        
        """
        Wrapper function that runs the COL detection algorithm and saves results to output files.

        Parameters
        ----------
        plt_or_not : int (0 or 1)
            If 1, generates and saves diagnostic plots of the algorithm.

        mod : str
            Name of the dataset or model (used for output file naming).

        ts : int
            Time step parameter (not explicitly used here but kept for consistency).

        thp : float
            Precipitation threshold used to select days for COL attribution.

        thf : float
            Additional threshold parameter (reserved for future use or consistency).

        dirout : str
            Output directory where result files and figures are saved.

        critere_taille : float
            Minimum spatial extent (in km²) required for a structure to be considered a COL.

        ste : float
            Step size used for scanning isohypses in the detection algorithm.

        numero : str
            Optional identifier appended to output filenames.

        lat_max, lat_min : float
            Latitude bounds defining the spatial domain of interest.

        lon_max, lon_min : float
            Longitude bounds defining the spatial domain of interest.

        test_mode : int (0 or 1)
            If 1, runs the algorithm on a single test date (d2p).

        d2p : datetime
            Specific test date used when test_mode = 1.

        criteria : float, optional
            Percentile used to define precipitation threshold.
            If None, the default percentile (self.qu) is used.

        Returns
        -------
        None
            Results are written to output text files:
            
            - Yearly statistics of detected events
            - Locations and dates of COL-attributed precipitation events
            - All detected COLs (including dry ones)
        """

        "Runs the COL detection algorithm and exports results"

        # ---------------------------------------------------------------------
        # Select relevant dates (either full dataset or single test case)
        # ---------------------------------------------------------------------

        ttp = self.time[(self.pr_ts > thp)]
        if test_mode and d2p is not None:
            ttp=self.time[(self.pr_ts > thp)&(self.time==d2p)]
           

        # ---------------------------------------------------------------------
        # Define output file names
        # ---------------------------------------------------------------------
        suf_sup = f"_{self.qu}_{lon_min}W_{lon_max}E_{lat_min}S_{lat_max}N"
        
        fn_out = f"{dirout}{mod}{numero}{suf_sup}.txt"
        fn_out1 = f"{dirout}{mod}_locs{numero}{suf_sup}.txt"
        fn_out2 = f"{dirout}{mod}_{numero}{suf_sup}_all_found_COLs_even_dry_ones.txt"

        print(fn_out)
        print(fn_out1)
        print(ttp)

        # ---------------------------------------------------------------------
        # Run main COL detection algorithm
        # ---------------------------------------------------------------------
        a, b, c, d, e, h, all_col_lons, all_col_lats, all_col_dates = self.col(
            dates=ttp[:],
            ste=ste,
            critere_taille=critere_taille,
            lat2=lat_max, lat1=lat_min,
            lon1=lon_min, lon2=lon_max,
            ff=dirout + 'forgif_' + mod + str(numero) + '/',
            plt_or_not=plt_or_not,
            thp=thp,
            criteria=criteria,
            showpl=showpl
        )

        # ---------------------------------------------------------------------
        # Save yearly statistics
        # ---------------------------------------------------------------------
        f = open(fn_out, mode='w')
        for i in range(a.shape[0]):
            f.write(str(a[i]) + '\t' + str(b[i]) + '\n')
        f.close()

        # ---------------------------------------------------------------------
        # Save detected COL locations and associated events
        # ---------------------------------------------------------------------
        f = open(fn_out1, mode='w')
        for i in range(c.shape[0]):
            f.write(
                '{:.2f}'.format(c[i]) + '\t' +
                '{:.2f}'.format(d[i]) + '\t' +
                e[i].strftime('%Y-%m-%d') + '\t' +
                str(h[i]) + '\n'
            )
        f.close()

        # ---------------------------------------------------------------------
        # Save all detected COLs (including non-precipitating ones)
        # ---------------------------------------------------------------------
        f = open(fn_out2, mode='w')
        for i in range(all_col_lats.shape[0]):
            f.write(
                '{:.2f}'.format(all_col_lons[i]) + '\t' +
                '{:.2f}'.format(all_col_lats[i]) + '\t' +
                all_col_dates[i].strftime('%Y-%m-%d') + '\n'
            )
        f.close()


    def close(self):
        """
        Safely releases memory by closing open datasets and deleting large attributes.

        This function is useful when working with large climate datasets to:
        - prevent memory leaks
        - free RAM after heavy computations
        - ensure proper closure of xarray datasets

        Notes
        -----
        - Attempts to close xarray datasets (`self.ds`, `self.dsp`)
        - Deletes large NumPy arrays stored as object attributes
        - Ignores missing attributes safely
        """
        
        # Fermer les fichiers ouverts par xarray
        try:
            self.ds.close()
        except:
            pass
        try:
            self.dsp.close()
        except:
            pass
    
        # Supprimer les grosses variables numpy
        attrs_to_delete = [
            "data", "datap", "maskp", "mask_map", "lons", "lats",
            "lons1", "lats1", "pr_ts", "pr_max", "pix_area"
        ]
    
        for a in attrs_to_delete:
            if hasattr(self, a):
                delattr(self, a)

    def TS_col_notcol_V2(self, direct='/srv5_tmp2/jbrajkovic/', mod='ERA5',
                  qu=90, qup=.9, thp=15, suf='_critaire_24h_1mm.txt',
                  fn_pr='', retm=0, lat_max=60, lat_min=-99,
                  lon_max=20, lon_min=-5, qum=0, decluster=0, r=3):
        """
        Extracts and compares precipitation time series associated with COL and non-COL events.

        This function separates precipitation events into:
        - COL-associated events (based on detected COL locations)
        - Non-COL events (remaining precipitation extremes)

        Parameters
        ----------
        direct : str
            Directory containing COL detection output files.

        mod : str
            Dataset/model name used for file naming.

        qu : float
            Percentile used in COL detection (for file naming consistency).

        qup : float
            Additional percentile parameter (reserved for extended analysis).

        thp : float
            Precipitation threshold (mm/day) used to define extreme events.

        suf : str
            Suffix used in output file naming.

        fn_pr : str
            Path to precipitation dataset (defaults to self.fnp).

        retm : int (0 or 1)
            If 1, returns spatial maps of precipitation statistics.

        lat_max, lat_min, lon_max, lon_min : float
            Optional spatial filtering applied to COL locations.

        qum : float
            Additional quantile parameter (reserved).

        decluster : int (0 or 1)
            If 1, applies temporal declustering to remove dependent events.

        r : int
            Minimum separation (in days) for declustering.

        Returns
        -------
        da_sel : ndarray
            Dates of COL-associated precipitation events.

        intg : ndarray
            Precipitation values associated with COL events.

        tinocol : ndarray
            Dates of non-COL precipitation events.

        prnocol : ndarray
            Precipitation values for non-COL events.

        tal, pral : ndarray
            Full (unfiltered) precipitation time series above threshold.

        gfh : ndarray (optional)
            Frequency map of precipitation exceeding threshold (if retm=1).

        mq : ndarray (optional)
            Mean precipitation map for exceedances (if retm=1).
        """
    
        print(direct)
        fn1 = direct + mod + '_locs_qu' + str(qu) + suf
        print('File read in function TS_col_notcol:', fn1)
    
        fn_pr = self.fnp
        pr_ts = self.pr_ts
        lons = self.lons1
    
        # Read COL locations and dates
        da1 = pd.read_csv(fn1, sep='\t', header=None)
        self.da1 = da1
    
        dates = pd.to_datetime(da1[2].values)
        da_pr = pd.to_datetime(ct.marray(self.dsp, 'time'))
    
        lons_gf = da1[0].values
        lats_gf = da1[1].values
    
        # Optional spatial filtering
        if lat_max != -99:
            mask_spat = (
                (lons_gf > lon_min) & (lons_gf < lon_max) &
                (lats_gf < lat_max) & (lats_gf > lat_min)
            )
            dates = dates[mask_spat]
            lons_gf = lons_gf[mask_spat]
            lats_gf = lats_gf[mask_spat]
    
        self.logf = np.zeros(0)
        self.lagf = np.zeros(0)
    
        Prs = np.zeros(dates.shape[0])
        intg = []
        da_sel = []
    
        qu /= 100
        t1 = da_pr.copy()
    
        Pr_list = []  # store precipitation fields, concatenate once at the end
    
        for d in range(dates.shape[0]):
            if d%100==0:
                print(f'{d} out of {dates.shape[0]}')
           
            Prs[d]=self.pr_ts[self.time==dates[d]]
    
            # Select COL events
            if Prs[d] > thp:
                intg.append(Prs[d])
                da_sel.append(dates[d])
      
        intg = np.array(intg)
        da_sel = pd.to_datetime(da_sel)
    
        # NO-COL events from full precipitation time series
        prnocol = pr_ts[pr_ts > thp]
        tinocol = t1[pr_ts > thp]
    
        # Optional temporal declustering
        if decluster:
            intg, da_sel = self.decluster(intg, da_sel, r=r)
            prnocol, tinocol = self.decluster(prnocol, tinocol, r=r)
            # print('I decluster with ')
    
        # Map effects for COL events
        if retm:
            dates_np = np.array(da_sel, dtype='datetime64')

            # Select all precipitation data at once
            # Pr_all.shape = (ntime, nlat, nlon)
            Pr_all = self.dsp['Precip'].sel(time=dates_np, method='nearest').values
            
            # Create mask where precip > threshold
            mask = Pr_all > thp  # shape: (ntime, nlat, nlon)
            
            # Count of exceedances at each grid point
            gfh = mask.sum(axis=0)
            
            # Sum of precipitation where threshold exceeded
            gfq = np.where(mask, Pr_all, 0).sum(axis=0)
            
            # Mean precip above threshold
            mq = np.full_like(gfh, np.nan, dtype=float)
            valid = gfh > 0
            mq[valid] = gfq[valid] / gfh[valid]
    
        pral = prnocol.copy()
        tal = pd.to_datetime(tinocol.copy())
    
        # Remove dates overlapping with COL events
        maska = np.isin(tinocol, da_sel)
        tinocol = tinocol[~maska]
        prnocol = prnocol[~maska]
    
        if retm:
            return da_sel, intg, tinocol, prnocol, tal, pral, gfh, mq
        else:
            return da_sel, intg, tinocol, prnocol, tal, pral   
        
    def decluster(self, precip_TS, dates_TS, r=3):
        """
        Applies temporal declustering to a precipitation time series.

        Consecutive events within a window of `r` days are grouped into clusters,
        and only the maximum precipitation event within each cluster is retained.

        Parameters
        ----------
        precip_TS : ndarray
            Time series of precipitation values.

        dates_TS : ndarray of datetime
            Corresponding dates of precipitation events.

        r : int
            Minimum separation (in days) required between independent events.

        Returns
        -------
        new_precip : ndarray
            Declustered precipitation values (cluster maxima).

        new_dates : ndarray of datetime
            Dates corresponding to declustered events.
        """
        
        new_dates=np.array([])
        new_precip=np.array([])
        
        i=0
        
        
        while i < len(precip_TS):
            candidat=precip_TS[i]
            candidat_date=dates_TS[i]
            
            if i!=len(precip_TS)-1:
                i1=i+1
                delta = dates_TS[i1]-candidat_date
                size_cluster=delta.total_seconds() / 86400
            else:
                i1=i+1
                size_cluster=1000
            
            while size_cluster<r and i1<precip_TS.shape[0]: 
                
                if precip_TS[i1]>candidat:
                    candidat = precip_TS[i1]
                    candidat_date = dates_TS[i1]
                i1+=1
                try:
                    delta = dates_TS[i1]-candidat_date
                    size_cluster=delta.total_seconds() / 86400
                except:
                    break
            i=i1
            
            new_dates=np.append(new_dates,candidat_date)
            new_precip=np.append(new_precip,candidat)
        new_dates=pd.to_datetime(new_dates)
        print(precip_TS.shape,new_precip.shape)
        
        print(dates_TS,new_dates)
        
        return new_precip,new_dates
            
 

 


if __name__ == "__main__":
    """
    Example script to run the COL detection algorithm on a given dataset.
    
    You can visualize and modify the algorithm parameters by using the sample datasets provided
    (sample_daily_precip.nc and sampe_daily_z500.nc)

    This script demonstrates how to:
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
        
    showplot : int (0 or 1)
        If 1, shows the plot at each algorithm step, if 0 the figures are saved i the ouput directory
    """

    mo = 'ERA5'
    ts = 24
    
    thp = 0
    cta = 50000
    qup = 99.9
    qu_rain=95
    showplot=0
    

    test_algo = 1

    "string to format output text files names"
    
    grid = '_ERA5grid'
    add_message = ''

        
    string_extension = f'_qu{qup:.0f}_area_{cta:.0f}km2_{ts:.0f}f_{thp}mm{grid}{add_message}'

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
                    criteria=95, 
                    showpl=showplot
                    )
    ec.close()
    del ec
    gc.collect()

    