# -*- coding: utf-8 -*-

from postSPIT import plotting_classes as plc
import os
from glob import glob
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from picasso.io import load_movie
from skimage.draw import polygon
#%%
##open analysis object the difficult way
directory = r'D:\Data\Chi_data\first data\output2\Run00002'
image0 = plc.Tracked_image(
                       load_movie(r'D:\Data\Chi_data\first data\output2\Run00002\Pat01_638nm.tif')[0],
                       pd.read_csv(directory + r'\Pat01_638nm_roi_locs_nm_trackpy.csv'), 
                       pd.read_hdf(directory + r'\Pat01_638nm_roi_locs_nm_trackpy_stats.hdf'),
                       load_movie(r'D:\Data\Chi_data\first data\output2\Run00002\Pat01_488nm.tif')[0], 
                       pd.read_csv(directory + r'\Pat01_488nm_roi_locs_nm_trackpy.csv'), 
                       pd.read_hdf(directory + r'\Pat01_488nm_roi_locs_nm_trackpy_stats.hdf'),
                       pd.read_csv(directory + r'\Pat01_638nm_roi_locs_nm_trackpy_colocsTracks.csv'),
                       pd.read_hdf(directory+r'\Pat01_638nm_roi_locs_nm_trackpy_colocsTracks_stats.hdf'), 
                       108                     
                       )
#%%
##open analysis object the easy way
directory = r'D:\Data\Chi_data\first data\output2\Run00002'
#(optional) You can check what is there analyzed in the folder with .validet()
plc.Single_tracked_folder(directory).validate()
# use .open_files() to open the analysis object
image1 = plc.Single_tracked_folder(directory).open_files()
#%%
##To plot, first explore interesting tracks. For that I would use the stats files:
ch0_stats = image1.stats0
ch1_stats = image1.stats1
coloc_stats = image1.coloc_stats
# now you can use the variable explorer of Spyder to get the track ID (or coloc track ID) of the track or tracks you want to plot. 
# The stats are open in a pd.DataFrame, so you cna always use your standard operation to filter track IDs as you want in  an automatic manner. 
# once you have the track or tarcks you want to plot you can plot them on top of an image (pass the tracks in a list!):   
image1.plot_tracks([1704], channel= 'ch0') #in here we plot from channel0 (the reference channel for colocalizing) the track number 1704.
image1.plot_tracks([2513], channel= 'ch1') # track 2513 for channel 1. 
image1.plot_colocs([248]) #in here we plot the coloc track (track in ch0, track in ch1, and coloc track on top of the reference channel image)

#%%
image1.intensity_coloc(470, 'upper left', 'CD19', 'Zap70') #in here we plot the intensities over time of a coloc track. We pass the track ID (not in a list now!)
                                                            # then as optional we can pass the location of the legend, and the labels for ch0 and ch1. 
#I still have to implement intensity plotting for a single channel (for one or multiple tracks)
#%%
#there is also the option to extract the diffusion coefficients. I will also implement extratcing dwell times in the future. Upon request, I
#can implement more data extraction. 
d = image1.extract_Ds(10, 'ch0')

#%%
#We also have operations for full datsets. For this, we first create a folder (the one we pass) and inside it we make subfolders named by the 
#condition. For example, in Chi's case it has 4 folders: CART3-HighExp, CART3-LowExp, CART4-HighExp, CART4-LowExp. Even if you only have one condition, 
#please, make a single subfolder 
#We set the object as:
a = plc.Dataset_tracked_folder(r'D:\Data\Chi_data\20250414\to_track2')
#%%
#we can seledct if we only want to work with a subset of the conditions. Migth be sometimes interesting to analyze only a subset of the data. 
a.select_conditions([0, 1])
#%%
#We can count the number of colocalized tracks. 
counting = a.count_number_cotracks()
#we can also aggregate the counting per condition. 
summary_counting = a.summary_count_number_cotracks()
#%%
#we can check which runs have colocalized tracks. If we have selected conditions, it will only do it for the selected ones. 
a.count_cotracked()
tracked_folders = a.tracked_folders
#%%
# I am currently working on extracting the diffusion coefficients of each run and have it saved per ROI and condition. The same for dwell times. 
# Let me know of any other request.   



