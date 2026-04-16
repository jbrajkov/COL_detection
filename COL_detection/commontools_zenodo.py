  #!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Dec  5 09:03:34 2022

@author: jbrajkovic
"""


import numpy as np


from scipy.spatial import cKDTree
from scipy.ndimage import label,binary_dilation

   
def clipping_data(data,mask_sp):
    "data structure = (t,y,x)"
    
    shi=0
    # print(mask_sp.shape,da_sh)
    
    # print('data shape in clipping_data:',data.shape)
    da_sh=np.array(data.shape)
    for i in range(mask_sp.shape[0]):

       sh=np.size(mask_sp[i,:][mask_sp[i,:]])
       if sh>0:
           shi+=1
           shj=sh
                    
    
    
    if np.size(da_sh)==2:
        new_da=np.zeros([shi,shj])
        
        suivi=0
        for i in range(data.shape[0]):
            sh=np.size(mask_sp[i,:][mask_sp[i,:]])
            if sh>0:
                new_da[suivi,:]=data[i,:][mask_sp[i,:]]
                suivi+=1
                
    elif np.size(da_sh)==3:
        new_da=np.zeros([data.shape[0],shi,shj])
        counti=-1
        for i in range(data.shape[1]):
            countj=0
            for j in range(data.shape[2]):   
                if mask_sp[i,j]:
                    # print(counti,countj)
                    new_da[:,counti,countj]=data[:,i,j]
                    
                    if countj==0:
                        counti+=1
                    countj+=1
    elif np.size(da_sh)==4:
         new_da=np.zeros([data.shape[0],data.shape[1],shi,shj])
         counti=-1
         for i in range(data.shape[2]):
             countj=0
             for j in range(data.shape[3]):   
                 if mask_sp[i,j]:
                     # print(counti,countj)
                     new_da[:,:,counti,countj]=data[:,:,i,j]
                     
                     if countj==0:
                         counti+=1
                     countj+=1
    data=np.array(new_da)
    
    return(data)

def find_closest_pixel_V2(lonsm, latsm, lonsi, latsi, neighbours=1):
    """
    Find the nearest pixels from an old grid to each point of a new grid.

    Parameters
    ----------
    lonsm : np.ndarray
        2D array of longitudes for the new grid.
    latsm : np.ndarray
        2D array of latitudes for the new grid.
    lonsi : np.ndarray
        2D array of longitudes for the old grid.
    latsi : np.ndarray
        2D array of latitudes for the old grid.
    neighbours : int, optional
        Number of nearest neighbours to return (default = 1).

    Returns
    -------
    output : np.ndarray
        4D array of shape (n_lat_new, n_lon_new, neighbours, 3) where:
          - output[..., n, 0] : latitude index on old grid
          - output[..., n, 1] : longitude index on old grid
          - output[..., n, 2] : weight (1 / distance_km)
    
    Notes
    -----
    - Uses a KDTree for fast nearest-neighbour search (greatly optimized).
    - Distance is computed in Euclidean space using (lat, lon) in degrees,
      but adjusted for approximate Earth curvature.
    - Weight = 1 / (distance_km + 1e-6) to avoid division by zero.
    """

    # Flatten old grid
    lat_flat = latsi.ravel()
    lon_flat = lonsi.ravel()
    
    # Build KDTree on old grid points
    tree = cKDTree(np.column_stack((lat_flat, lon_flat)))

    # Flatten new grid for query
    lat_new = latsm.ravel()
    lon_new = lonsm.ravel()

    # Query k nearest neighbours
    dist, idx = tree.query(np.column_stack((lat_new, lon_new)), k=neighbours)
    
    # Ensure shape consistency
    if neighbours == 1:
        dist = dist[:, np.newaxis]
        idx = idx[:, np.newaxis]

    # Convert 1D indices back to 2D (old grid)
    i_idx, j_idx = np.unravel_index(idx, lonsi.shape)

    # Compute weights (inverse distance, avoid division by zero)
    weights = 1.0 / np.maximum(dist * 111.0, 1e-6)  # ~km assuming degrees*111km

    # Reshape to (nlat_new, nlon_new, neighbours, 3)
    nlat_new, nlon_new = lonsm.shape
    output = np.empty((nlat_new, nlon_new, neighbours, 3), dtype=float)
    output[..., 0] = i_idx.reshape(nlat_new, nlon_new, neighbours)
    output[..., 1] = j_idx.reshape(nlat_new, nlon_new, neighbours)
    output[..., 2] = weights.reshape(nlat_new, nlon_new, neighbours)

    return output


def interpolation_grid_2d(data,
                          lons_new_grid=np.zeros(0),
                          lats_new_grid=np.zeros(0),
                          lons_old_grid=np.zeros(0),
                          lats_old_grid=np.zeros(0),
                          neighbours=4):
    """
    Interpolates 2D data from an old latitude/longitude grid onto a new grid
    using a weighted average of the nearest neighbours.

    Parameters
    ----------
    data : np.ndarray
        2D array of data values on the old grid with shape (n_lat_old, n_lon_old).
    lons_new_grid : np.ndarray
        2D array of longitudes for the new grid.
    lats_new_grid : np.ndarray
        2D array of latitudes for the new grid.
    lons_old_grid : np.ndarray
        2D array of longitudes for the old grid.
    lats_old_grid : np.ndarray
        2D array of latitudes for the old grid.
    neighbours : int, optional
        Number of nearest neighbours to use for interpolation (default = 1).

    Returns
    -------
    returned_matrix : np.ndarray
        2D array of interpolated data with shape (n_lat_new, n_lon_new).

    Notes
    -----
    - The interpolation is based on weighted averaging using distances
      returned by `find_closest_pixel`.
    """
    
    # Find indices and weights of closest neighbours
    neigh = find_closest_pixel_V2(lons_new_grid, lats_new_grid,
                                  lons_old_grid, lats_old_grid,
                                  neighbours=neighbours)
    
    nlat_new, nlon_new = neigh.shape[:2]
    returned_matrix = np.zeros((nlat_new, nlon_new))
    
    # Extract neighbour info once
    i_idx = neigh[..., 0].astype(int)
    j_idx = neigh[..., 1].astype(int)
    weights = neigh[..., 2]
    weight_total = np.sum(weights, axis=-1)
    
    # Extract neighbor values and compute weighted sum
    vals = data[i_idx, j_idx]
    weighted_sum = np.sum(vals * weights, axis=-1)
    returned_matrix[...] = weighted_sum / np.maximum(weight_total, 1e-12)
    
    return returned_matrix

def make_hist(vect,vmin=0,vmax=5,step=1):
    'returns frequencies and associated center of classes'
    bounds=np.arange(vmin,vmax+step,step)
    count=np.zeros(bounds.shape[0]-1)
    if np.size(vect.shape)>1:
        vect1=vect.flatten()
    else:
        vect1=np.array(vect)
    for v in vect1:
        for b in range(bounds.shape[0]-1):
            if v>=bounds[b]and v<bounds[b+1]:
                count[b]+=1
                break
    probs=count/float(vect1.shape[0])
    centers=bounds[1:]-step/2
    return(probs,centers)
    

def delimitate_closed_zones_optimized(mat, vmax=10, vmin=0, mas1=None, nmax=None):
    """
    Détecte les zones fermées et isolées dans `mat` entre vmin et vmax,
    sans contact avec les bords. Renvoie une matrice avec les zones numérotées.
    """
    if mas1 is None:
        mas1 = np.ones_like(mat, dtype=bool)
        mas1[0, :] = False
        mas1[-1, :] = False
        mas1[:, 0] = False
        mas1[:, -1] = False

    # Créer un masque des zones à détecter (valeurs comprises entre vmin et vmax)
    zone = (mat >= vmin) & (mat <= vmax) & mas1

    # Étiqueter les régions connectées
    labeled, num_features = label(zone)

    # Résultat initial
    result = np.full_like(mat, np.nan, dtype=float)
    val_clust = 1

    for label_id in range(1, num_features + 1):
        if nmax is not None and val_clust > nmax:
            break

        region = (labeled == label_id)

        # Vérifie si la région touche un bord
        if np.any(region[0, :]) or np.any(region[-1, :]) or np.any(region[:, 0]) or np.any(region[:, -1]):
            continue  # la zone touche un bord → rejetée

        # Vérifie si elle est entièrement entourée de valeurs hors [vmin, vmax]
        border = binary_dilation(region) & (~region)
        if not np.all((mat[border] < vmin) | (mat[border] > vmax)):
            continue  # pas complètement fermée → rejetée

        # Zone validée
        result[region] = val_clust
        val_clust += 1
        # fig=plt.figure(),plt.imshow(result)
    return result

def dis2pix(lat1,lon1,lat2,lon2):
    """returns dist in meters"""
    lat1,lon1,lat2,lon2=np.deg2rad(lat1),np.deg2rad(lon1),np.deg2rad(lat2),np.deg2rad(lon2)
    dis=np.arccos(np.sin(lat1)*np.sin(lat2)+np.cos(lat1)*np.cos(lat2)*np.cos(abs(lon1-lon2)))*6371000
    return(dis)

def marray(ds, var):
    """Return NumPy array for variable `var`, with last dimension first."""
    return np.asarray(ds[var]).T