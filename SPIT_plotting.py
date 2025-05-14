# -*- coding: utf-8 -*-
import os
from glob import glob
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from picasso.io import load_movie
from skimage.draw import polygon
#plot single track on image 

def plot_tracks_by_id(movie, stats, csv, to_check, px2nm):
    """
    Creates an image where one or multiple tracks are shown on top of the maximum projection of the image. 
    
    Input: 
        movie: video used to create maximum projection
        stats: hdf file generated after tracking with SPIT open in a pandas dataframe.
        csv: csv file generated after tracking with SPIT open in a pandas dataframe. 
        to_check: list containing the index of the tracks to show.
        px2nm: pixel size in nm. 
        
    Output:
        fig: matplotlib figure object
    """
    print('Tracks inside ROIs:')
    print(stats[stats['track.id'].isin(to_check)]['cell_id'].unique())
    
    fig, ax = plt.subplots()
    
    if 'contour' in stats.columns and len(stats[stats['track.id'].isin(to_check)]['cell_id'].unique()) == 1:
        contour = stats[stats['track.id'] == to_check[0]]['contour'].values[0]
        x0 = min(contour[:, 0])
        x1 = max(contour[:, 0])
        y0 = min(contour[:, 1])
        y1 = max(contour[:, 1])
        ax.imshow(np.max(movie, axis=0)[y0:y1, x0:x1], cmap='gray')
        for i in to_check:
            ax.plot(csv[csv['track.id'] == i].x/px2nm - x0, csv[csv['track.id'] == i].y/px2nm - y0)
    else: 
        ax.imshow(np.max(movie, axis=0), cmap='gray')
        for i in to_check:
            ax.plot(csv[csv['track.id'] == i].x/px2nm, csv[csv['track.id'] == i].y/px2nm)
    
    ax.grid(False)
    ax.axis('off')
    
    return fig


#plot coloc tracks from colocsTracks files

def plot_coloc_by_id(movie, stats, csv, to_check, px2nm):
    print('Tracks inside ROIs:')
    print(stats[stats['colocID'].isin(to_check)]['cell_id'].unique())
    print()
    if 'contour' in stats.columns and len(stats[stats['colocID'].isin(to_check)]['cell_id'].unique()) == 1:
        cantour = stats[stats['colocID'] == to_check[0]]['contour'].values[0]
        x0 = min(cantour[:, 0])
        x1 = max(cantour[:, 0])
        y0 = min(cantour[:, 1])
        y1 = max(cantour[:, 1])
        plt.figure()
        plt.imshow(np.max(movie, axis = 0)[y0:y1, x0:x1], cmap='gray')
        for i in to_check:
            plt.plot(csv[csv['colocID'] == i].x_0/px2nm - x0, csv[csv['colocID'] == i].y_0/px2nm - y0, color = 'red')
            plt.plot(csv[csv['colocID'] == i].x_1/px2nm - x0, csv[csv['colocID'] == i].y_1/px2nm - y0, color = 'blue')
            plt.plot(csv[csv['colocID'] == i].x/px2nm - x0, csv[csv['colocID'] == i].y/px2nm - y0, color = 'purple')
    else: 
        plt.figure()
        plt.imshow(np.max(movie, axis = 0), cmap='gray')
        for i in to_check:
            plt.plot(csv[csv['colocID'] == i].x_0/px2nm, csv[csv['colocID'] == i].y_0/px2nm, color = 'red')
            plt.plot(csv[csv['colocID'] == i].x_1/px2nm, csv[csv['colocID'] == i].y_1/px2nm, color = 'blue')
            plt.plot(csv[csv['colocID'] == i].x/px2nm, csv[csv['colocID'] == i].y/px2nm, color = 'purple')
    plt.grid(False)
    plt.axis('off')
    plt.show()
    plt.close()

#plot intensity profile of track from original version of SPIT
def intensity_coloc_old(moviech0, locsch0, track_id, moviech1=None, locsch1=None, df_coloc_csv=None):

    def create_mask(image_shape, contour):
        mask = np.ones(image_shape, dtype=bool)
        rr, cc = polygon(contour[:, 1], contour[:, 0], image_shape)
        mask[rr, cc] = False
        return mask
    if df_coloc_csv is not None:
        to_check = track_id
        plt.figure()
    
        locID0 = df_coloc_csv[df_coloc_csv['track.id'] == to_check]['locID0'].tolist()
        tracks488 = locsch0[locsch0['locID'].isin(locID0)]['track.id'].tolist()
        locs488 = locsch0[locsch0['track.id'].isin(tracks488)][['x', 'y', 't', 'intensity', 'track.id']]
        locs488['xpx'] = locs488.x/108
        locs488['ypx'] = locs488.y/108
        
        locID0 = df_coloc_csv[df_coloc_csv['track.id'] == to_check]['locID1'].tolist()
        tracks638 = locsch1[locsch1['locID'].isin(locID0)]['track.id'].tolist()
        locs638 = locsch1[locsch1['track.id'].isin(tracks638)][['x', 'y', 't', 'intensity', 'track.id']]
        
        locs488['im_int'] = None
        image_size = moviech0[0].shape
        # mask = create_mask(image_size, cantour)
        for index, row in locs488.iterrows():
            # print(movie[row.t].shape)    
            locs488.at[index, 'im_int'] = np.mean(moviech0[row.t, int(row.y/108)-1:int(row.y/108)+1, int(row.x/108)-1:int(row.x/108)+1])/np.median(moviech0[row.t])
        locs638['im_int'] = None
        for index, row in locs638.iterrows():
            locs638.at[index, 'im_int'] = np.mean(moviech1[row.t, int(row.y/108)-1:int(row.y/108)+1, int(row.x/108)-1:int(row.x/108)+1])/np.median(moviech1[row.t])
        plt.plot(locs488.t*2, (locs488.im_int).rolling(window=1).mean(), label='ch0') #/max(locs488.intensity)
        plt.plot(locs638.t*2, (locs638.im_int).rolling(window=1).mean(), label='ch1') #/max(locs638.intensity)
        plt.axvline(x=min(df_coloc_csv[df_coloc_csv['track.id'] == to_check].t)*2, color='r', linestyle='--', label='start colocalization')
        plt.axvline(x=max(df_coloc_csv[df_coloc_csv['track.id'] == to_check].t)*2, color='r', linestyle='--', label='end colocalization')
        plt.legend()
        plt.xlabel('Time(sec)')
        plt.ylabel('Pixel intensity/median \n bacgkrond intensity (AU)')
        plt.show()
        plt.close()
    else: 
        plt.figure()
        locs488 = locsch0[locsch0['track.id'].isin([track_id])][['x', 'y', 't', 'intensity', 'track.id']]
        locs488['xpx'] = locs488.x/108
        locs488['ypx'] = locs488.y/108
        locs488['im_int'] = None
        for index, row in locs488.iterrows():
            # print(movie[row.t].shape)    
            locs488.at[index, 'im_int'] = np.mean(moviech0[row.t, int(row.y/108)-1:int(row.y/108)+1, int(row.x/108)-1:int(row.x/108)+1])/np.median(moviech0[row.t])
        plt.plot(locs488.t*2, (locs488.im_int).rolling(window=1).mean(), label='ch0') #/max(locs488.intensity)
        plt.legend()
        plt.xlabel('Time(sec)')
        plt.ylabel('Pixel intensity/median \n bacgkrond intensity (AU)')
        plt.show()
        plt.close()
#plot intensity profile of colocalized track from new version of SPIT
def intensity_coloc(csv, movie0, movie1, to_check, px2nm):
    # plt.figure()
    intensity0 = []
    intensity1 = []
    track = csv[csv.colocID == to_check].copy()
    track['int_im_0'] = None
    track['int_im_1'] = None
    for index, row in track.iterrows(): 
        if not np.isnan(row.y_0):
            track.at[index, 'im_int_0'] =  np.mean(movie0[row.t, int(row.y_0/px2nm)-1:int(row.y_0/px2nm)+1, int(row.x_0/px2nm)-1:int(row.x_0/px2nm)+1])/np.median(movie0[row.t])
        if not np.isnan(row.y_1):
            track.at[index, 'im_int_1'] = np.mean(movie1[row.t, int(row.y_1/px2nm)-1:int(row.y_1/px2nm)+1, int(row.x_1/px2nm)-1:int(row.x_1/px2nm)+1])/np.median(movie1[row.t])
    plt.plot(track.t*2, (track.im_int_0).rolling(window=1).mean(), label='ch0') #/max(locs488.intensity)
    plt.plot(track.t*2, (track.im_int_1).rolling(window=1).mean(), label='ch1') #/max(locs638.intensity)
    colocalized_df = track[(track['distance'] <= 300) & pd.notna(track['x']) & pd.notna(track['y'])]
    if not colocalized_df.empty:
        start_time = colocalized_df['t'].min()*2
        end_time = colocalized_df['t'].max()*2
    plt.axvline(x=start_time, color='r', linestyle='--', label='start colocalization')
    plt.axvline(x=end_time, color='r', linestyle='--', label='end colocalization')
    plt.legend()
    plt.xlabel('Time(sec)')
    plt.ylabel('Pixel intensity/median \n bacgkrond intensity (AU)')   
    plt.show()
    plt.close()
        
def dwell_times(folder, groups):
    #To_od: modify to allow different combinations of channels
    """
    Computes dwell times from a ColocsTracks output files. Dwell times will be calculated for reference channel (0) set during 5.Coloc_tracks.py
    
    Input: 
        folder: string with the folder to analyze (can contain subfolders)
        input: list of strings conatining the groups to be extracted from the subfolder name. If any string in the list is in the subfolder name, 
                that string will be saved as the group
    Outupt:
        dwell_times: pandas dataframe containing 7 columns:
            ColocID
            Track_id_0
            Track_id_1
            cell_id
            dwell_time
            folder (subfolder analyzed, if it is emtpy: the file used for the calculation was found in the input folder)
            group 
    """
    def extract_group(run_name, groups):
        if any(substring in run_name for substring in groups):
            for i in groups:
                if i in run_name:
                    return i
        else: 
            return np.NAN
    data = []
    directory_path = folder 
    pathscsv = glob(directory_path + '/**/**.csv', recursive=True)
    paths_locs = list(set(os.path.dirname(file) for file in pathscsv))
    dict_hdf = {}
    dict_csv = {}
    for image in tqdm(paths_locs, desc = 'Opening files...'):
        if os.path.isdir(image):
            run = image.replace(directory_path, '')
            pathcsv638 = glob(image + '/**/**colocsTracks.csv', recursive=True)
            pathhdf = glob(image + '/**/**colocsTracks_stats.hdf', recursive=True)
            if pathhdf:
                dict_hdf[run] = [pd.read_hdf(pathhdf[0])]
                dict_csv[run] = [pd.read_csv(pathcsv638[0])]
    
    for run  in tqdm(dict_csv.keys(), desc = 'Computing dwell times...'):
        hdf_coloc = dict_hdf[run][0]
        final_df = dict_csv[run][0]
        unique_colocIDs = hdf_coloc[(hdf_coloc['num_frames_coloc'] >10)]['colocID'].unique()
        for i in unique_colocIDs:#[7:8]:
            colocalized_df = final_df[(final_df['colocID'] == i) & (final_df['distance'] <= 300) & pd.notna(final_df['x']) & pd.notna(final_df['y'])]
            if not colocalized_df.empty:
                start_time = colocalized_df['t'].min()*2
            locs_ch0 = final_df[final_df.colocID == i][['x_0', 'y_0', 't', 'intensity_0']]
            locs_ch0 = locs_ch0.dropna(axis = 0)
            locs_ch1 = final_df[final_df.colocID == i][['x_1', 'y_1', 't', 'intensity_1']]
            locs_ch1 = locs_ch1.dropna(axis = 0)
            times_0 = locs_ch0.t
            if min(times_0) * 2 < start_time:
                dt = start_time - min(times_0) * 2
                
                # Create a Series with the same columns
                data.append([i, hdf_coloc[hdf_coloc.colocID == i]['track.id0'].values[0], 
                                      hdf_coloc[hdf_coloc.colocID == i]['track.id1'].values[0], 
                                      hdf_coloc[hdf_coloc.colocID == i]['cell_id'].values[0],
                                      dt, 
                                      run])
                
                dwell_times = pd.DataFrame(data, columns=['colocID', 'track.id0', 'track.id1', 'cell_id', 'dwell_time', 'run'])
    dwell_times['group'] = dwell_times['run'].apply(extract_group, args = (groups, ))
    return dwell_times

def hist_dwell(dw, to_plot, density, ylim = None, xstart = 0):
    plt.figure()
    for group in to_plot:
        subset = dw[dw['group'] == group]
        plt.hist(subset['dwell_time'], density = density, bins=20, alpha=0.5, label=group)
    
    plt.xlabel('Dwell Time')
    plt.ylabel('Frequency')
    plt.xlim(xstart)
    plt.ylim(0, ylim)
    plt.legend()
    plt.show()

def diff_coefs(folder, groups, ch, min_len):
    """
    Extracts diffusion coefficient of particles in a specific channel.
    
    Input: 
        folder: string with the folder to analyze (can contain subfolders)
        groups: list of strings containing the groups to be extracted from the subfolder name. If any string in the list is in the subfolder name, 
                that string will be saved as the group
        ch: channel from which you want the D_msds
    Output:
        Ds: pandas dataframe containing 5 columns:
            Track_id
            cell_id
            D
            folder (subfolder analyzed, if it is empty: the file used for the calculation was found in the input folder)
            group 
    """
    def extract_group(run_name, groups):
        if any(substring in run_name for substring in groups):
            for i in groups:
                if i in run_name:
                    return i
        else: 
            return np.NAN

    data = []
    directory_path = folder 
    pathscsv = glob(directory_path + '/**/**.hdf', recursive=True)
    paths_locs = list(set(os.path.dirname(file) for file in pathscsv))
    dict_hdf = {}
    
    for image in tqdm(paths_locs, desc='Opening files...'):
        if os.path.isdir(image):
            run = image.replace(directory_path, '')
            pathhdf = glob(f"{image}/**/*_{ch}nm*_locs_nm_trackpy_stats.hdf", recursive=True)
            if pathhdf:
                dict_hdf[run] = [pd.read_hdf(pathhdf[0])]
    
    all_Ds = pd.DataFrame()
    columns_to_extract = ['track.id', 'cell_id', 'D_msd']
    
    for run in tqdm(dict_hdf.keys(), desc='Extracting D_msds...'):
        hdf_coloc = dict_hdf[run][0]
        Ds = hdf_coloc[(hdf_coloc['length'] >= min_len)][columns_to_extract]
        Ds = Ds.dropna()
        Ds['run'] = run
        all_Ds = pd.concat([all_Ds, Ds], ignore_index=True)
    
    all_Ds['group'] = all_Ds['run'].apply(extract_group, args=(groups,))
    
    #TODO: Create a new column 'group_cell_ID' where each cell is called from 0 until n within each group
    all_Ds['group_cell_ID'] = all_Ds.groupby('group').cumcount()
    
    return all_Ds

def box_D_condition(Ds, to_plot, ylim=None):
    plt.figure()
    filtered_Ds = Ds[Ds['group'].isin(to_plot)]
    boxplot = filtered_Ds.boxplot(column='D_msd', by='group', showfliers=False, boxprops=dict(color='black'), 
                                  medianprops=dict(color='black'), whiskerprops=dict(color='black'),capprops=dict(color='black'))

    # Overlay datapoints with jitter
    for i, group in enumerate(to_plot):
        group_data = filtered_Ds[filtered_Ds['group'] == group]
        jitter = 0.1 * (np.random.rand(len(group_data)) - 0.5)
        plt.scatter(np.full(len(group_data), i + 1) + jitter, group_data['D_msd'], color='red', alpha=0.5, s=2)

    # Remove all gridlines
    plt.grid(False)
    
    plt.xlabel('')
    plt.suptitle('')
    plt.ylabel('D_msd (um^2/sec)')
    plt.title('')
    if ylim:
        plt.ylim(0, ylim)

    plt.show()

def box_D_cell(Ds, to_plot, ylim=None):
    #TODO: STILL WORKING ON IT###
    plt.figure()
    # Create a new column 'cell' by combining 'run' and 'cell_id'
    Ds['cell'] = Ds['run'].astype(str) + '_' + Ds['cell_id'].astype(str)
    
    filtered_Ds = Ds[Ds['group'].isin(to_plot)]
    boxplot = filtered_Ds.boxplot(column='D_msd', by='cell', showfliers=False,
                                  boxprops=dict(color='black', linewidth=2), medianprops=dict(color='black'), 
                                  whiskerprops=dict(color='black'), capprops=dict(color='black'))

    # Overlay datapoints with jitter
    cells = filtered_Ds['cell'].unique()
    for i, cell in enumerate(cells):
        cell_data = filtered_Ds[filtered_Ds['cell'] == cell]
        jitter = 0.1 * (np.random.rand(len(cell_data)) - 0.5)
        plt.scatter(np.full(len(cell_data), i + 1) + jitter, cell_data['D_msd'], color='red', alpha=0.5, s=5)

    # Remove all gridlines
    plt.grid(False)
    
    plt.xlabel('')
    plt.suptitle('')
    plt.ylabel('D_msd (um^2/sec)')
    plt.title('')
    if ylim:
        plt.ylim(0, ylim)

    # Rotate x-axis labels to vertical
    plt.xticks(rotation=90)

    plt.show()
#%%
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
fig = plot_tracks_by_id(movie488, stats_488, df_488, [2513,2376, 309 ], 108)
# fig.savefig(r'D:\Data\my_plot.pdf') 
#%%
plot_coloc_by_id(movie488, df_coloc_hdf, df_coloc_csv, [74], 108)
#%%
intensity_coloc(df_coloc_csv, movie488, movie638, 74, 108)


#%%
groups = ['CART3-LowExp', 'CART4-LowExp', 'CART3-HighExp', 'CART4-HighExp']
Ds = diff_coefs(r'D:\Data\Chi_data\20250414\to_track2', groups, 488, 10)
to_plot = ['CART4-LowExp', 'CART4-HighExp']
box_D_condition(Ds, to_plot)


#%%
groups = ['CART3-LowExp', 'CART4-LowExp', 'CART3-HighExp', 'CART4-HighExp']
dw = dwell_times(r'D:\Data\Chi_data\20250414\to_track2', groups)
#%%
to_plot = ['CART3-LowExp', 'CART4-LowExp']
hist_dwell(dw, to_plot, True, None, 0)


