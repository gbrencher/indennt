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
def hyp3_to_ds(path):
    '''
    Reads unwrapped phase, coherence, and DEM into xarray dataset from single hyp3 folder 
    '''
    # globs for data to load
    unw_phase_path = glob(f'{path}/*unw_phase.tif')[0]
    dem_path = glob(f'{path}/*dem.tif')[0]

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

# function to prepare arrays for model run
def arrays_to_tensor(ds, norm=True, igram_norm=[-41, 41]):
    
    # interpolate nans (will crash model otherwise)
    unw_phase_ds = ds.unw_phase.interpolate_na(dim='x', use_coordinate=False)
    unw_phase_ds = unw_phase_ds.interpolate_na(dim='y', use_coordinate=False)
    
    # set remaining nans to 0 and convert to tensor
    igram_tensor = torch.Tensor(unw_phase_ds.to_numpy()).nan_to_num(0)
    dem_tensor = torch.Tensor(ds.elevation.to_numpy()).nan_to_num(0)
    
    # normalize input images for best results
    if norm==True:
        igram_tensor = 2*(((igram_tensor-igram_norm[0])/(igram_norm[1]-igram_norm[0])))-1
        dem_tensor = 2*(((dem_tensor-dem_tensor.min())/(dem_tensor.max()-dem_tensor.min())))-1
    
    return igram_tensor, dem_tensor

#function to return to original values
def undo_norm(array, min=-41, max=41):
    array = ((array+1)*((max-min)/2))+min
    return array

# tiled prediction to avoid large RAM usage
def tiled_prediction(ds, igram, dem, model, tile_size=1024):
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
    noise = undo_norm(noise)
    signal = undo_norm(signal)
    
    # inherit nans from original interferogram
    noise[ds.unw_phase.isnull()] = np.nan
    signal[ds.unw_phase.isnull()] = np.nan

    return noise, signal

# correct a single hyp3 igram
def correct_single_igram(granule_path, model):
    ds = hyp3_to_ds(granule_path)
    igram, dem = arrays_to_tensor(ds)
    noise, signal = tiled_prediction(ds=ds, igram=igram, dem=dem, model=model)

    ds['pred_noise'] = (('y', 'x'), noise)
    ds['pred_signal'] = (('y', 'x'), signal)
    
    return ds

# correct all hyp3 igrams in a directory
def correct_hyp3_dir(hyp3_path, model, skip_exist=True):
    hyp3_list = os.listdir(hyp3_path)
    for i, granule in enumerate(hyp3_list):
        granule_path = f'{hyp3_path}/{granule}'
        if skip_exist==True:
            if os.path.exists(f'{granule_path}/{granule}_unw_phase_CNN_signal.tif'):
                print(f'unw_phase_CNN already in {granule}, skipping') 
                continue
        print(f'working on {granule}, {i+1}/{len(hyp3_list)}')
        
        ds = hyp3_to_ds(granule_path)
        igram, dem = arrays_to_tensor(ds)
        noise, signal = tiled_prediction(ds=ds, igram=igram, dem=dem, model=model)

        ds['pred_noise'] = (('y', 'x'), noise)
        ds['pred_signal'] = (('y', 'x'), signal)

        ds.pred_noise.rio.to_raster(f'{granule_path}/{granule}_unw_phase_CNN_noise.tif')
        ds.pred_signal.rio.to_raster(f'{granule_path}/{granule}_unw_phase_CNN_signal.tif')
    
    