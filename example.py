# -*- coding: utf-8 -*-

from postSPIT import plotting as pl
import os
from glob import glob
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from picasso.io import load_movie
from skimage.draw import polygon

directory = r'D:\Data\Chi_data\first data\output2\Run00002'
df_coloc_hdf = pd.read_hdf(directory+r'\Pat01_638nm_roi_locs_nm_trackpy_colocsTracks_stats.hdf')
df_coloc_csv = pd.read_csv(directory + r'\Pat01_638nm_roi_locs_nm_trackpy_colocsTracks.csv')
nm488 = directory + r'\Pat01_488nm_roi_locs_nm_trackpy.csv'
df_488 =pd.read_csv(nm488)
stats_488 = pd.read_hdf(directory + r'\Pat01_488nm_roi_locs_nm_trackpy_stats.hdf')
nm638 = directory + r'\Pat01_638nm_roi_locs_nm_trackpy.csv'
df_638 =pd.read_csv(nm638)
stats_638 = pd.read_hdf(directory + r'\Pat01_638nm_roi_locs_nm_trackpy_stats.hdf')
movie488, info= load_movie(r'D:\Data\Chi_data\first data\output2\Run00002\Pat01_488nm.tif')
movie638, info= load_movie(r'D:\Data\Chi_data\first data\output2\Run00002\Pat01_638nm.tif')
#%%
fig = pl.plot_tracks_by_id(movie488, stats_488, df_488, [2513,2376, 309 ], 108)
# fig.savefig(r'D:\Data\my_plot.pdf') 
#%%
pl.plot_coloc_by_id(movie488, df_coloc_hdf, df_coloc_csv, [74], 108)
#%%
pl.intensity_coloc(df_coloc_csv, movie488, movie638, 74, 108)


#%%
groups = ['CART3-LowExp', 'CART4-LowExp', 'CART3-HighExp', 'CART4-HighExp']
Ds = pl.diff_coefs(r'D:\Data\Chi_data\20250414\to_track2', groups, 488, 10)
to_plot = ['CART4-LowExp', 'CART4-HighExp']
pl.box_D_condition(Ds, to_plot)


#%%
groups = ['CART3-LowExp', 'CART4-LowExp', 'CART3-HighExp', 'CART4-HighExp']
dw = pl.dwell_times(r'D:\Data\Chi_data\20250414\to_track2', groups)
#%%
to_plot = ['CART3-LowExp', 'CART4-LowExp']
pl.hist_dwell(dw, to_plot, True, None, 0)


