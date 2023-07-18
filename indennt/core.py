import xarray as xr
from glob import glob
import rasterio as rio
import rioxarray
import torch
import torch.nn.functional as F
import numpy as np
import os
import math

# load in single igram and other data 
def hyp3_to_ds(path, igram_suffix, dem_suffix):
    '''
    Reads unwrapped phase and DEM into xarray dataset from single hyp3 folder 
    '''
    # globs for data to load
    unw_phase_path = glob(f'{path}/*{igram_suffix}')[0]
    dem_path = glob(f'{path}/*{dem_suffix}')[0]

    # list granules for coordinate
    granule = os.path.split(unw_phase_path)[-1][0:-14]

    # read unw_phase into data array and assign coordinates
    da = xr.open_dataset(unw_phase_path)
    da = da.assign_coords({'granule':('granule', [granule])})
    
    # concatenate into dataset and rename variable
    ds = da.rename({'band_data': 'unw_phase'})

    #open dem into datasets
    dem_ds = xr.open_dataset(dem_path)

    # add dem to unw_phase dataset
    ds['elevation'] = (('band', 'y', 'x'), dem_ds.band_data.values)

    # remove band coordinate
    ds = ds.squeeze()

    return ds

def isce_to_ds(path):
    '''
    Reads unwrapped phase and DEM into xarray dataset from single isce folder 
    '''
    # globs for data to load
    phase_path = glob(f'{path}/merged/filt_topophase.unw.geo.vrt')[0]
    dem_path = glob(f'{path}/merged/dem.crop.vrt')[0]
    
    # read unw_phase into data array
    ds = rioxarray.open_rasterio(phase_path, masked=True).isel(band=1).to_dataset(name='unw_phase')
    ds = ds.where(ds['unw_phase'] != 0.0) # isce uses 0 for nodata
      
    #open dem and add to ds
    dem_da = rioxarray.open_rasterio(dem_path)
    ds['elevation'] = (('y', 'x'), dem_da.squeeze().values)
    
    # reproject to utm
    ds = ds.rio.reproject(ds.rio.estimate_utm_crs())
    
    return ds

# function to prepare arrays for model run
def arrays_to_tensor(ds, igram_norm, dem_norm, use_igram_range, use_dem_range, norm=True):
    
    # interpolate nans (will crash model otherwise)
    unw_phase_ds = ds.unw_phase.interpolate_na(dim='x', use_coordinate=False)
    unw_phase_ds = unw_phase_ds.interpolate_na(dim='y', use_coordinate=False)
    
    # set remaining nans to 0 and convert to tensor
    igram_np = unw_phase_ds.to_numpy()
    dem_np = ds.elevation.to_numpy()
    
    # normalize input images for best results
    if norm==True:
        if use_igram_range==True:
            igram_np = 2*(((igram_np-igram_np.min())/(igram_np.max()-igram_np.min())))-1
        else:
            igram_np = 2*(((igram_np-igram_norm[0])/(igram_norm[1]-igram_norm[0])))-1
        if use_dem_range==True:
            dem_np = 2*(((dem_np-dem_np.abs().min())/(dem_np.max()-dem_np.abs().min())))-1
        else:
            dem_np = 2*(((dem_np-dem_norm[0])/(dem_norm[1]-dem_norm[0])))-1
            
    # set remaining nans to 0 and convert to tensor
    igram_tensor = torch.Tensor(igram_np).nan_to_num(0)
    dem_tensor = torch.Tensor(dem_np).nan_to_num(0)
    
    return igram_tensor, dem_tensor

#function to return to original values
def undo_norm(array, igram_norm):
    array = ((array+1)*((igram_norm[1]-igram_norm[0])/2))+igram_norm[0]
    return array

# tiled prediction to avoid large RAM usage
def tiled_prediction(ds, igram, dem, model, igram_norm, tile_size=1024):
    xmin=0
    xmax=tile_size
    ymin=0
    ymax=tile_size
    
    # pad left and bottom 
    igram_pad = F.pad(igram, (0, tile_size, 0, tile_size), 'constant', 0)
    dem_pad = F.pad(dem, (0, tile_size, 0, tile_size), 'constant', 0)
    noise_pad = np.empty_like(dem_pad.numpy())

    #loop through tiles
    for i in range(math.ceil((len(ds.x)/tile_size))):
        #print(f'column {i}')
        for j in range(math.ceil((len(ds.y)/tile_size))):
            #print(f'row {j}')
            ymin = j*tile_size
            ymax = (j+1)*tile_size
            xmin = i*tile_size
            xmax = (i+1)*tile_size
            
            # predict noise in tile
            with torch.no_grad():
                noise = model(igram_pad[None, None, ymin:ymax, xmin:xmax], dem_pad[None, None, ymin:ymax, xmin:xmax])
            noise_pad[ymin:ymax, xmin:xmax] = noise.detach().squeeze().numpy()
            
    # recover original dimensions
    noise = noise_pad[0:(len(ds.y)), 0:(len(ds.x))]
    # correct interferogram
    signal = igram.squeeze().numpy() - noise
    
    # undo normalization
    noise = undo_norm(noise, igram_norm)
    signal = undo_norm(signal, igram_norm)
    
    # inherit nans from original interferogram
    noise[ds.unw_phase.isnull()] = np.nan
    signal[ds.unw_phase.isnull()] = np.nan

    return noise, signal

# correct a single hyp3 igram
def correct_single_igram(igram_path,
                         model,
                         processor,
                         igram_suffix='unw_phase.tif',
                         dem_suffix='dem.tif',
                         igram_norm=[-41, 41],
                         dem_norm=[0, 4400],
                         use_igram_range = False,
                         use_dem_range=False
                        ):

    if processor == 'hyp3':
        ds = hyp3_to_ds(igram_path, igram_suffix, dem_suffix)
    if processor == 'isce':
        ds = isce_to_ds(igram_path)
    igram, dem = arrays_to_tensor(ds, igram_norm, dem_norm, use_igram_range, use_dem_range)
    noise, signal = tiled_prediction(ds=ds, igram=igram, dem=dem, model=model, igram_norm=igram_norm)

    ds['pred_noise'] = (('y', 'x'), noise)
    ds['pred_signal'] = (('y', 'x'), signal)
    
    return ds

# correct all hyp3 igrams in a directory
def correct_igram_dir(path,
                      model,
                      processor,
                      igram_suffix='unw_phase.tif',
                      dem_suffix='dem.tif',
                      igram_norm=[-41, 41],
                      dem_norm=[0, 4400],
                      use_igram_range = False,
                      use_dem_range=False,
                      skip_exist=True
                     ):
    
    igram_list = os.listdir(path)
    for i, igram_name in enumerate(igram_list):
        igram_path = f'{path}/{igram_name}'
        if skip_exist==True:
            if os.path.exists(f'{igram_path}/{igram_name}_unw_phase_CNN_signal.tif'):
                print(f'unw_phase_CNN already in {igram_name}, skipping') 
                continue
        
        print(f'working on {igram_name}, {i+1}/{len(igram_list)}')
        if processor == 'hyp3':
            ds = hyp3_to_ds(igram_path, igram_suffix, dem_suffix)
        if processor == 'isce':
            ds = isce_to_ds(igram_path)
            
        igram, dem = arrays_to_tensor(ds, igram_norm, dem_norm, use_igram_range, use_dem_range)
        noise, signal = tiled_prediction(ds=ds, igram=igram, dem=dem, model=model, igram_norm=igram_norm)
        
        ds['pred_noise'] = (('y', 'x'), noise)
        ds['pred_signal'] = (('y', 'x'), signal)

        ds.pred_noise.rio.to_raster(f'{igram_path}/{igram_name}_unw_phase_CNN_noise.tif')
        ds.pred_signal.rio.to_raster(f'{igram_path}/{igram_name}_unw_phase_CNN_signal.tif')