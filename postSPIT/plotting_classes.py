from picasso.io import TiffMultiMap, load_movie
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.stats import ttest_ind
from tqdm import tqdm
import os
from natsort import natsorted
from glob import glob
import yaml
from spit import tools
import re
from skimage.draw import polygon
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening, binary_closing, disk, remove_small_objects, remove_small_holes
import cv2
from skimage.filters import (
    threshold_otsu,
    threshold_yen,
    threshold_triangle,
    threshold_li,
    threshold_mean,
    threshold_minimum,
    threshold_isodata, 
    threshold_local
)
import tifffile
import trackpy as tp
import logging
from collections import defaultdict
from scipy.ndimage import median_filter
from scipy.stats import linregress
from skimage.feature import local_binary_pattern
logging.getLogger('trackpy').setLevel(logging.ERROR)


class Plotter:
    def __init__(self, title="Plot", xlabel="X-axis", ylabel="Y-axis", figsize = (6.4, 4.8)):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.fig, self.ax = plt.subplots(figsize=figsize)
    def set_labels(self):
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)     
    def set_ylim(self, s, e = None):
        self.ax.set_ylim(s, e)
    def set_xlim(self, s, e = None):
        self.ax.set_xlim(s, e)
    def show_plot(self, legend_loc='best'):
        '''
        Shows plot with the legend (if there is any). Locations for the legend:
            'best' (default): Automatically chooses the best location.
            'upper right'
            'upper left'
            'lower left'
            'lower right'
            'right'
            'center left'
            'center right'
            'lower center'
            'upper center'
        '''
        
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(loc=legend_loc)
        plt.show()
        return self.fig
    def set_grid(self, visible=True, which='both', axis='both', linestyle='--', linewidth=0.5):
        self.ax.grid(visible, which=which, axis=axis, linestyle=linestyle, linewidth=linewidth)
    def save_plot(self, filename, dpi = 300):
        self.fig.savefig(filename,dpi = dpi, bbox_inches='tight', pad_inches=0.1)
    def refresh(self, legend_loc='best'):
        plt.ion()  # Ensure interactive mode is on
        plt.figure(self.fig.number)  # Reactivate the figure
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(loc=legend_loc)
        self.fig.canvas.draw()  # Redraw the canvas
        self.fig.canvas.flush_events()  # Ensure updates are shown

class LinePlotter(Plotter):
    def add_data(self, x, y, label="Line", color="blue"):
        self.ax.plot(x, y, label=label, color=color)

class ScatterPlotter(Plotter):
    def add_data(self, x, y, label="Scatter", color="green"):
        self.ax.scatter(x, y, label=label, color=color)

class BoxPlotter(Plotter):
    def __init__(self, title="Box Plot", xlabel="X-axis", ylabel="Y-axis"):
        super().__init__(title, xlabel, ylabel)
        self.data = []
        self.labels = []
    def add_box(self, data, label=None):
        self.data.append(data)
        if label:
            self.labels.append(label)
        else:
            self.labels.append(f"Group {len(self.data)}")
        self.ax.clear()  # Clear the previous plot
        self.ax.boxplot(self.data, labels=self.labels, showfliers=False)
        self.set_labels()
        self.add_jittered_dots()
    def add_jittered_dots(self):
        for i, data in enumerate(self.data):
            x = np.random.normal(i + 1, 0.04, size=len(data))
            self.ax.plot(x, data, 'r.', alpha=0.5)
    def add_statistical_annotations(self):
        num_groups = len(self.data)
        y_max = max([max(group) for group in self.data])
        y_min = min([min(group) for group in self.data])
        y_range = y_max - y_min
        y_offset = y_range * 0.1

        for i in range(num_groups):
            for j in range(i + 1, num_groups):
                t_stat, p_val = ttest_ind(self.data[i], self.data[j])
                x1, x2 = i + 1, j + 1
                y = y_max + y_offset * (j - i)
                self.ax.plot([x1, x1, x2, x2], [y, y + y_offset, y + y_offset, y], lw=1.5, c='k')
                self.ax.text((x1 + x2) * 0.5, y + y_offset, f"p = {p_val:.3e}", ha='center', va='bottom')

class HistogramPlotter(Plotter):
    def __init__(self,title="Histogram", xlabel="X-axis", ylabel="Y-axis", figsize=(6.4, 4.8)):
        super().__init__(xlabel=xlabel, ylabel=ylabel, figsize=figsize)
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # default matplotlib colors
        self.color_index = 0

    def add_data(self, data, bins=50, label="Histogram", alpha=0.5):
        color = self.colors[self.color_index % len(self.colors)]
        self.color_index += 1
        self.ax.hist(data, bins=bins, label=label, color=color, alpha=alpha)

class TrackPlotter(Plotter):
    def __init__(self, image, px2nm=108, title="Track Plot", xlabel="X (nm)", ylabel="Y (nm)", figsize = (6.4, 6.4)):
        super().__init__(title, xlabel, ylabel, figsize)
        self.image = image
        self.px2nm = px2nm
        self.ax.set_aspect('equal')
    def show_max_projection(self, contour=None):
        img = np.max(self.image, axis=0)
        if contour is not None:
            x0, x1 = int(min(contour[:, 0])), int(max(contour[:, 0]))
            y0, y1 = int(min(contour[:, 1])), int(max(contour[:, 1]))
            img = img[y0:y1, x0:x1]
        self.ax.imshow(img, cmap='gray')
        return (x0 if contour is not None else 0), (y0 if contour is not None else 0)
    def plot_tracks(self, df, tid, id_col='track.id', x_col='x', y_col='y', color='blue', offset=(0, 0)):
        track = df[df[id_col] == tid]
        self.ax.plot(track[x_col]/self.px2nm - offset[0], track[y_col]/self.px2nm - offset[1], color=color)
        self.ax.grid(False)
        self.ax.axis('off')
    def save_plot(self, filename, dpi = 300):
        self.fig.savefig(filename,dpi = dpi, bbox_inches='tight', pad_inches=0)

class Tracked_image:
    def __init__(self, ch0, tracks0, stats0, ch1=None, tracks1=None, stats1=None, coloc_tracks=None, coloc_stats=None, nm2px=108, folder = None):
        #checking data types
        assert isinstance(ch0, TiffMultiMap), "'ch0' must be a picasso.io.TiffMultiMap object"
        assert isinstance(tracks0, pd.DataFrame), "'tracks0' must be a pd.DataFrame object"
        assert isinstance(stats0, pd.DataFrame), "'stats0' must be a pd.DataFrame object"
        assert isinstance(nm2px, (int, float))
        # Optional checks
        if ch1 is not None:
            assert isinstance(ch1, TiffMultiMap), "'ch1' must be a picasso.io.TiffMultiMap object"
        if tracks1 is not None:
            assert isinstance(tracks1, pd.DataFrame), "'tracks1' must be a pandas DataFrame"
        if stats1 is not None:
            assert isinstance(stats1, pd.DataFrame), "'stats1' must be a pandas DataFrame"
        if coloc_tracks is not None:
            assert isinstance(coloc_tracks, pd.DataFrame), "'coloc_tracks' must be a pandas DataFrame"
        if coloc_stats is not None:
            assert isinstance(coloc_stats, pd.DataFrame), "'coloc_stats' must be a pandas DataFrame"
        
        self.ch0 = ch0
        self.ch1 = ch1
        self.tracks0 = tracks0
        self.stats0 = stats0
        self.tracks1 = tracks1
        self.stats1 = stats1
        self.coloc_tracks = coloc_tracks
        self.coloc_stats = coloc_stats
        self.nm2px = nm2px
        self.folder = folder
        self.result_cluster_analysis = {'ch0' : None, 'ch1' : None}
        self.summary_cluster_analysis = {'ch0' : None, 'ch1' : None}
        self.linked_clusters = {'ch0' : None, 'ch1' : None}
        self.linked_clusters_stats = {'ch0' : None, 'ch1' : None}
        self.tracks_outside_clusters = {'ch0' : None, 'ch1' : None}
    def plot_tracks(self, to_check, channel='ch0'):
        if channel == 'ch0':
            image = self.ch0
            tracks = self.tracks0
            stats = self.stats0
        elif channel == 'ch1':
            assert self.ch1 is not None, "ch1 is not provided."
            assert self.tracks1 is not None, "tracks1 is not provided."
            assert self.stats1 is not None, "stats1 is not provided."
            image = self.ch1
            tracks = self.tracks1
            stats = self.stats1
        else:
            raise ValueError("channel must be 'ch0' or 'ch1'")
    
        plotter = TrackPlotter(image, self.nm2px)
        print('Tracks inside ROIs:', stats[stats['track.id'].isin(to_check)]['cell_id'].unique())
        
        contour = None
        if 'contour' in stats.columns and len(stats[stats['track.id'].isin(to_check)]['cell_id'].unique()) == 1:
            contour = stats[stats['track.id'] == to_check[0]]['contour'].values[0]
        
        offset = plotter.show_max_projection(contour)
        for i in to_check:
            if channel == 'ch0':
                plotter.plot_tracks(tracks, i, offset=offset, color='red')
            elif channel == 'ch1':
                plotter.plot_tracks(tracks, i, offset=offset, color='blue')
        plotter.show_plot()
        return plotter   
    def plot_colocs(self, to_check):
        assert self.coloc_tracks is not None, "coloc_tracks is not provided."
        assert self.coloc_stats is not None, "coloc_stats is not provided."
    
        plotter = TrackPlotter(self.ch0, self.nm2px)
        print('Tracks inside ROIs:', self.coloc_stats[self.coloc_stats['colocID'].isin(to_check)]['cell_id'].unique())
        
        contour = None
        if 'contour' in self.coloc_stats.columns and len(self.coloc_stats[self.coloc_stats['colocID'].isin(to_check)]['cell_id'].unique()) == 1:
            contour = self.coloc_stats[self.coloc_stats['colocID'] == to_check[0]]['contour'].values[0]
        
        offset = plotter.show_max_projection(contour)
        for i in to_check:
            plotter.plot_tracks(self.coloc_tracks, i,id_col = 'colocID', x_col='x_0', y_col='y_0', color='red', offset=offset)
            plotter.plot_tracks(self.coloc_tracks, i,id_col = 'colocID', x_col='x_1', y_col='y_1', color='blue', offset=offset)
            plotter.plot_tracks(self.coloc_tracks, i,id_col = 'colocID', x_col='x', y_col='y', color='purple', offset=offset)
        plotter.show_plot()
        return plotter
    def intensity_coloc(self, to_check, legend_loc = 'best', legend_0 = 'ch0', legend_1 = 'ch1'):
        """
        Creates a plot of the intensity of two colocalizing tracks over time, with vertical dash lines indicating where they start and stop colocalizing.
        Only usable with coloc_tracks pipeline.

        Parameters:
            to_check (int): colocID of the track to plot.
        """
        assert self.coloc_tracks is not None, "coloc_tracks is not provided."
        assert self.ch1 is not None, "ch1 is not provided."

        plotter = LinePlotter(title="Intensity Over Time", xlabel="Time (sec)", ylabel="Spot intensity/median BG intensity (AU)")

        track = self.coloc_tracks[self.coloc_tracks.colocID == to_check].copy()
        track['im_int_0'] = None
        track['im_int_1'] = None

        for index, row in track.iterrows():
            if not np.isnan(row.y_0):
                track.at[index, 'im_int_0'] = np.mean(self.ch0[row.t, int(row.y_0/self.nm2px)-1:int(row.y_0/self.nm2px)+1, int(row.x_0/self.nm2px)-1:int(row.x_0/self.nm2px)+1]) / np.median(self.ch0[row.t])
            if not np.isnan(row.y_1):
                track.at[index, 'im_int_1'] = np.mean(self.ch1[row.t, int(row.y_1/self.nm2px)-1:int(row.y_1/self.nm2px)+1, int(row.x_1/self.nm2px)-1:int(row.x_1/self.nm2px)+1]) / np.median(self.ch1[row.t])

        plotter.add_data(track.t * 2, track.im_int_0.rolling(window=1).mean(), label=legend_0, color='blue')
        plotter.add_data(track.t * 2, track.im_int_1.rolling(window=1).mean(), label=legend_1, color='red')

        colocalized_df = track[(track['distance'] <= 300) & pd.notna(track['x']) & pd.notna(track['y'])]
        if not colocalized_df.empty:
            start_time = colocalized_df['t'].min() * 2
            end_time = colocalized_df['t'].max() * 2
            plotter.ax.axvline(x=start_time, color='black', linestyle='--', label='colocalization')
            plotter.ax.axvline(x=end_time, color='black', linestyle='--')
        plotter.set_labels()
        plotter.set_grid()
        plotter.set_ylim(0)
        plotter.show_plot(legend_loc= legend_loc)
        return plotter
    def extract_Ds(self, min_len=10, channel='ch0'):
        if channel == 'ch0':
            stats = self.stats0
        elif channel == 'ch1':
            if self.stats1 is None:
                # Return empty DataFrame with expected columns
                return None
            else:
                stats = self.stats1
        else:
            raise ValueError("channel must be 'ch0' or 'ch1'")
    
        columns_to_extract = ['track.id', 'cell_id', 'D_msd']
        ds = stats[stats['length'] >= min_len][columns_to_extract]
        ds = ds.dropna(subset=['D_msd'])
        return ds
    def extract_dwell(self, frame_rate = 1, min_len=10, max_dist = 250, ref='ch0'):
        stats = self.coloc_stats
        tracks = self.coloc_tracks
        if all(isinstance(obj, pd.DataFrame) for obj in [stats, tracks]):
            unique_colocIDs = stats[(stats['num_frames_coloc'] >min_len)]['colocID'].unique()
            data = []
            for i in unique_colocIDs:#[7:8]:
                colocalized_df = tracks[(tracks['colocID'] == i) & (tracks['distance'] <= max_dist) & pd.notna(tracks['x']) & pd.notna(tracks['y'])]
                if not colocalized_df.empty:
                    start_time = colocalized_df['t'].min()*frame_rate
                locs_ch0 = tracks[tracks.colocID == i][['x_0', 'y_0', 't', 'intensity_0']]
                locs_ch0 = locs_ch0.dropna(axis = 0)
                locs_ch1 = tracks[tracks.colocID == i][['x_1', 'y_1', 't', 'intensity_1']]
                locs_ch1 = locs_ch1.dropna(axis = 0)
                times_0 = {'ch0': locs_ch0.t, 'ch1': locs_ch1.t}.get(ref)
                if min(times_0) * frame_rate < start_time:
                    dt = start_time - min(times_0) * frame_rate
                    # Create a Series with the same columns
                    other_ref = 'ch1' if ref == 'ch0' else 'ch0'
                    track_ids = {'ch0': stats[stats.colocID == i]['track.id0'].values[0], 'ch1': stats[stats.colocID == i]['track.id1'].values[0]}
                    data.append([i, track_ids.get(ref), track_ids.get(other_ref),
                                          
                                          stats[stats.colocID == i]['cell_id'].values[0],
                                          dt])
                    
            dwell_times = pd.DataFrame(data, columns=['colocID', 'track.id_ref', 'track.id_binds', 'cell_id', 'dwell_time'])
            
            return dwell_times
        else:
            return None
    def split_cells(self):
        #obtain unique contours
        roi_contours = []
        roi_centroids = []
        if self.folder:
            rois = natsorted(glob(os.path.join(self.folder+"/*.roi")))  
            for roi in rois:
                    roi_contour = tools.get_roi_contour(roi)
                    roi_centroid = tools.get_roi_centroid(roi_contour)
                    roi_contours.append(roi_contour)
                    roi_centroids.append(roi_centroid)
            unique_contours = pd.Series(roi_contours)
            unique_centroids = pd.Series(roi_centroids)
        else:
            unique_contours = self.stats0.loc[self.stats0['contour'].apply(lambda x: str(x)).drop_duplicates().index, 'contour']
            unique_centroids = self.stats0.loc[self.stats0['centroid'].apply(lambda x: str(x)).drop_duplicates().index, 'centroid']

        
        #initilize variable to save output
        self.sep_cells = []
        self.contours = []
        self.centroids = []
        #for each contour
        for i, j in zip(unique_contours, unique_centroids):
            #extract the bounding box (the min and max in each direction)
            x0, x1 = int(min(i[:, 0])), int(max(i[:, 0]))
            y0, y1 = int(min(i[:, 1])), int(max(i[:, 1]))
            #append a list containing each of the bounding-bix cropped images 
            # print(x0, x1, y0, y1)
            self.sep_cells.append([self.ch0[:, y0:y1, x0:x1], self.ch1[:, y0:y1, x0:x1]])
            #correct the contour by substracting x0 and y0, then append it into the list
            corr = i.copy()
            corr[:, 0] -= x0
            corr[:, 1] -= y0
            corr_cen = j.copy()
            corr_cen[:, 0] -= x0
            corr_cen[:, 1] -= y0
            self.contours.append(corr)
            self.centroids.append(corr_cen)
    def analyze_clusters_protein(self, min_size = 80,ch='ch0', th_method = 'li_local', global_th_mode = 'max', window_size = 15, p = 2, q= 6, filter_spots = True, save_videos = False):
        """
        Analyze protein clusters per cell by thresholding and tracking features across frames.
    
        Parameters:
        -----------
        min_size : int
            Minimum size of detected cluster regions.
        ch : str
            Channel to analyze: 'ch0' or 'ch1'.
        save_videos : bool
            Whether to generate debug videos per cell.
    
        Returns:
        --------
        clusters_binary : list of 3D np.array
            Binary masks of detected clusters.
        result : pd.DataFrame
            Region properties per frame.
        linked_df : pd.DataFrame
            Tracked cluster features across frames.
        """
        self.split_cells()
        clusters_binary = []
        all_props = []
        self.cluster_contours = {}  # Store contours by cell/frame/label

        for i in range(len(self.sep_cells)):
            if ch == 'ch0':
                to_work = self.sep_cells[i][0]
            elif ch == 'ch1':
                to_work = self.sep_cells[i][1]
            # to_work = median_filter(to_work, size=(3, 1, 1))
            
            if  'li' in th_method:
                global_mask = self._li_threshold(to_work, self.contours[i], mode=global_th_mode)
            elif 'otsu' in th_method:
                global_mask = self._otsu_threshold(to_work, self.contours[i], mode=global_th_mode)
            else:
                raise ValueError(f'{th_method} is not a valid thresholding method')
            
            if 'local' in th_method: 
                local_mask = self._phansalkar_threshold(to_work, radius=window_size, p = p, q = q)
                binary_stack = self._remove_small_objects_per_frame(binary_opening((binary_closing(local_mask & global_mask))), min_size = min_size)
            else:
                binary_stack = global_mask
            clusters_binary.append(binary_stack)
            mask = self._create_mask(to_work[0].shape, self.contours[i])
            if i not in self.cluster_contours:
                self.cluster_contours[i] = {}

            for frame_num, binary_frame in enumerate(binary_stack):
                if frame_num not in self.cluster_contours[i]:
                    self.cluster_contours[i][frame_num] = {}

                labeled = label(binary_frame)

                for region in regionprops(labeled, intensity_image=to_work[frame_num]):
                    if region.area <= min_size:
                        continue

                    region_mask = (labeled == region.label).astype(np.uint8)
                    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if len(contours) == 0:
                        contour_list = np.array([]).reshape(0, 2)
                        perimeter = np.nan
                        solidity = np.nan
                        area = 0
                    else:
                        cnt = max(contours, key=cv2.contourArea)
                        perimeter = cv2.arcLength(cnt, True)
                        contour_list = cnt.reshape(-1, 2)
                        area = cv2.contourArea(cnt)
                        hull_area = cv2.contourArea(cv2.convexHull(cnt))
                        solidity = area / hull_area if hull_area > 0 else np.nan

                    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter and perimeter > 1e-6 else np.nan

                    M = cv2.moments(region_mask)
                    if M['m00'] != 0:
                        centroid_row = M['m01'] / M['m00']
                        centroid_col = M['m10'] / M['m00']
                    else:
                        centroid_row, centroid_col = region.centroid

                    bbox = region.bbox
                    region_mask_bool = (labeled == region.label)
                    median_int_out = np.median(to_work[frame_num][~mask])
                    median_int = np.median(to_work[frame_num][region_mask_bool])
                    sum_int = np.sum(to_work[frame_num][region_mask_bool])
                    
                    # Additional shape features
                    minr, minc, maxr, maxc = bbox
                    bbox_height = maxr - minr
                    bbox_width = maxc - minc
                    aspect_ratio = region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else np.nan
                    extent = region.area / ((bbox_height) * (bbox_width)) if bbox_height > 0 and bbox_width > 0 else np.nan
                    eccentricity = region.eccentricity if hasattr(region, 'eccentricity') else np.nan

                    props = {
                        'cell_id': i,
                        'frame': frame_num,
                        'area': area * self.nm2px * self.nm2px,
                        'sum_int': sum_int,
                        'norm_sum_int': sum_int / median_int_out,
                        'med_int': median_int,
                        'norm_med_int': median_int / median_int_out,
                        'dist_cent': np.linalg.norm(np.array([centroid_row, centroid_col]) - self.centroids[i][0]) * self.nm2px,
                        'solidity': solidity,
                        'perimeter': perimeter * self.nm2px,
                        'circularity': circularity,
                        'centroid_row': centroid_row,
                        'centroid_col': centroid_col,
                        'bbox_min_row': bbox[0],
                        'bbox_min_col': bbox[1],
                        'bbox_max_row': bbox[2],
                        'bbox_max_col': bbox[3],
                        'contour': contour_list, 
                        'label': region.label,
                        'aspect_ratio': aspect_ratio,
                        'extent': extent,
                        'eccentricity': eccentricity
                    }
                    # props.update(lbp_features)
                    all_props.append(props)

                    # Store contour by label for this frame & cell
                    self.cluster_contours[i][frame_num][region.label] = contour_list

        result = pd.DataFrame(all_props)
        results_stats = self._summarize_clusters_per_cell_frame(result)
        df_tp = result.rename(columns={"centroid_col": "x", "centroid_row": "y", "norm_sum_int": "mass", "area": "size"})
        df_tp['label'] = result['label']

        linked_all = []
        for cell_id, df_cell in df_tp.groupby('cell_id'):
            linked = tp.link_df(df_cell, search_range=25, memory=0, adaptive_step=0.95,
            adaptive_stop = 2, link_strategy='hybrid')
            linked['cell_id'] = cell_id
            linked_all.append(linked)
            
        linked_df = pd.concat(linked_all, ignore_index=True)
        linked_stats = self._summarize_per_track(linked_df)
        self.result_cluster_analysis[ch] = result
        self.summary_cluster_analysis[ch] = results_stats
        self.linked_clusters[ch] = linked_df
        self.linked_clusters_stats[ch] = linked_stats
        spots = None
        if filter_spots:
            spots = self.combine_clusters_and_spots(channel = ch)
        if save_videos:
            self._save_centroid_videos_per_cell(clusters_binary, result, self.sep_cells, ch=ch, square_size=1, filtered_spots = spots)
        
        return clusters_binary, result, results_stats, linked_df, linked_stats
    def combine_clusters_and_spots(self, channel = 'ch0'):
        #TODO: fix issue with ROIs: if one ROI does not have tracks, the rois are not properly assigned. --> probably fix in SPIT.link by making the cell_id columns have the correct roi number. 
        clusters = self.result_cluster_analysis[channel].copy()
        if channel == 'ch0':
            spots = self.tracks0.copy()
        elif channel == 'ch1':
            spots = self.tracks1.copy()
        else:
            print(f'Channel {channel} is not valid. Use ch0 or ch1.')
            return 
        cells = list(spots.loc[spots['cell_id'].apply(lambda x: str(x)).drop_duplicates().index, 'cell_id'])
        unique_contour = self.stats0.loc[self.stats0['contour'].apply(lambda x: str(x)).drop_duplicates().index, 'contour']
        corrections = {}
        for cell, contour in zip(cells, unique_contour):
            x0 = int(min(contour[:, 0]))
            y0 = int(min(contour[:, 1]))
            corrections[cell] = (x0, y0)
        spots[['x_per_cell', 'y_per_cell']] = spots.apply(
            lambda row: pd.Series([
                row['x']/self.nm2px - corrections[row['cell_id']][0],  
                row['y']/self.nm2px - corrections[row['cell_id']][1]   ]),
            axis=1
            )
        # clusters['contour_shifted'] = clusters.apply(
        #         lambda row: row['contour'] + np.array([corrections[row['cell_id']][1], corrections[row['cell_id']][0]]),  # row, col
        #         axis=1
        #                 )
        filtered_spots = self._filter_spots_outside_clusters(spots, clusters)
        
        
        self.tracks_outside_clusters[channel] = filtered_spots
        return filtered_spots
    def _filter_spots_outside_clusters(self, spots_df, clusters_df):
        # Create a mask to keep track of which spots to keep
        keep_mask = []
        
        for idx, spot in spots_df.iterrows():
            frame = spot['t']
            # print(int(frame))
            x, y = spot['x_per_cell'], spot['y_per_cell']
            # Get all contours for this frame
            contours = clusters_df[clusters_df['frame'] == int(frame)]['contour']
            
            # Check if the point is inside any contour
            inside_any = False
            for contour in contours:
                if contour.size == 0:
                    continue
                result = cv2.pointPolygonTest(contour, (x, y), False)
                if result >= 0:  # >0 inside, =0 on edge, <0 outside
                    inside_any = True
                    break

            # Keep the spot only if it's not inside any contour
            keep_mask.append(not inside_any)

        # Filter the spots
        filtered_spots = spots_df[keep_mask].reset_index(drop=True)
        return filtered_spots
    def _li_threshold(self, image, mask, mode = "max"):
        if mode == 'max':
            thresh = threshold_li(np.max(image, axis = 0))
        elif mode == 'full':
            thresh = threshold_li(np.array(image))
        elif mode == 'median':
            thresh = threshold_li(np.median(image, axis = 0))
        elif mode == 'last':
            thresh = threshold_li(image[-1])
        else:
            print("mode is not valid")
            return None
        global_mask = binary_closing((image > thresh) & self._create_mask(image[0].shape, mask))
        return global_mask
    def _otsu_threshold(self, original, image, mask, mode = "max"):
        if mode == 'max':
            thresh = threshold_otsu(np.max(original, axis = 0))
        elif mode == 'full':
            thresh = threshold_otsu(np.array(original))
        elif mode == 'median':
            thresh = threshold_otsu(np.median(original, axis = 0))
        elif mode == 'last':
            thresh = threshold_otsu(original[-1])
        else:
            print("mode is not valid")
            return None
        global_mask = binary_closing((image > thresh) & self._create_mask(image[0].shape, mask))
        return global_mask
    def _phansalkar_threshold(self, image_stack, radius=15, k=0.25, p=2.0, q=10.0):
        """
        Phansalkar local thresholding.
        Can process either a single 2D image or a 3D stack of images.
        
        Parameters:
            image_stack: 2D or 3D numpy array (float32, normalized [0,1])
            radius: int, local window radius
            k, p, q: floats, Phansalkar parameters
        Returns:
            Binary thresholded image or stack of same shape (bool ndarray)
        """
        window_size = (radius * 2) + 1

        def threshold_single(image):
            image = image / np.max(image) if np.max(image) > 0 else image
            mean = cv2.blur(image, (window_size, window_size))
            mean_sq = cv2.blur(image**2, (window_size, window_size))
            std = np.sqrt(mean_sq - mean**2)
            threshold = mean * (1 + p * np.exp(-q * mean) + k * ((std / 0.5) - 1))
            return image > threshold

        if image_stack.ndim == 2:
            return threshold_single(image_stack)
        elif image_stack.ndim == 3:
            # Process each frame independently and stack results
            binary_stack = np.zeros_like(image_stack, dtype=bool)
            for i in range(image_stack.shape[0]):
                binary_stack[i] = threshold_single(image_stack[i])
            return binary_stack
        else:
            raise ValueError("Input must be 2D or 3D numpy array")
    def _remove_small_objects_per_frame(self, stack, min_size=100, connectivity=1):
        cleaned_stack = np.zeros_like(stack, dtype=bool)
        for i in range(stack.shape[0]):  # assuming frames on axis 0
            cleaned_stack[i] = remove_small_objects(stack[i], min_size=min_size, connectivity=connectivity)
        return cleaned_stack
    def _summarize_clusters_per_cell_frame(self, result_df):
        summary_rows = []

        for (cell_id, frame), group in result_df.groupby(['cell_id', 'frame']):
            weights = group['area'] * group['norm_med_int']
            summary = {
                'cell_id': cell_id,
                'frame': frame,
                'num_clusters': len(group),
                'area_mean': self._weighted_mean(group['area'], weights),
                'area_safe_std': self._safe_std(group['area']),
                'sum_int_mean': self._weighted_mean(group['sum_int'], weights),
                'sum_int_safe_std': self._safe_std(group['sum_int']),
                'norm_sum_int_mean': self._weighted_mean(group['norm_sum_int'], weights),
                'norm_sum_int_safe_std': self._safe_std(group['norm_sum_int']),
                'med_int_mean': self._weighted_mean(group['med_int'], weights),
                'med_int_safe_std': self._safe_std(group['med_int']),
                'norm_med_int_mean': self._weighted_mean(group['norm_med_int'], weights),
                'norm_med_int_safe_std': self._safe_std(group['norm_med_int']),
                'dist_cent_mean': self._weighted_mean(group['dist_cent'], weights),
                'dist_cent_safe_std': self._safe_std(group['dist_cent']),
                'solidity_mean': self._weighted_mean(group['solidity'], weights),
                'solidity_safe_std': self._safe_std(group['solidity']),
                'perimeter_mean': self._weighted_mean(group['perimeter'], weights),
                'perimeter_safe_std': self._safe_std(group['perimeter']),
                'circularity_mean': self._weighted_mean(group['circularity'], weights),
                'circularity_safe_std': self._safe_std(group['circularity']),
                'aspect_ratio_mean': self._weighted_mean(group['aspect_ratio'], weights),
                'aspect_ratio_safe_std': self._safe_std(group['aspect_ratio']),
                'eccentricity_mean': self._weighted_mean(group['eccentricity'], weights),
                'eccentricity_safe_std': self._safe_std(group['eccentricity']),
                'extent_mean': self._weighted_mean(group['extent'], weights),
                'extent_safe_std': self._safe_std(group['extent']),
            }
            summary_rows.append(summary)

        return pd.DataFrame(summary_rows) 
    def _weighted_mean(self, x, weights):
        return np.average(x, weights=weights) if len(x) > 0 and np.sum(weights) > 0 else np.nan
    def _safe_std(self, x):
        return x.std() if len(x) > 1 else 0
    def _summarize_per_track(self, linked_df, min_frames=5):
        features = []
        
        for (cell_id, particle), group in linked_df.groupby(['cell_id', 'particle']):
            group = group.sort_values('frame')
            if len(group) < min_frames:
                continue
    
            frames = group['frame'].values
            # Convert coordinates to nanometers
            x = group['x'].values * self.nm2px
            y = group['y'].values * self.nm2px
            dists_to_center = group['dist_cent'].values  # already in nm if set that way
            
            # Compute frame-to-frame displacements and instantaneous speeds
            dx = np.diff(x)
            dy = np.diff(y)
            disp = np.sqrt(dx**2 + dy**2)  # instantaneous displacement (nm)
            
            total_distance = np.sum(disp)
            net_displacement = np.linalg.norm([x[-1] - x[0], y[-1] - y[0]])
            directionality_ratio = net_displacement / total_distance if total_distance > 0 else 0
            # TODO: convert frame difference to actual time if needed
            duration = frames[-1] - frames[0]
            speed = total_distance / duration if duration > 0 else 0  # mean speed (nm per frame)
    
            # Radial regression from dists_to_center vs frames (slope in nm per frame)
            slope, intercept, r_value, p_value, std_err = linregress(frames, dists_to_center)
    
            # Compute instantaneous angles for movement
            angles = np.arctan2(dy, dx)
            # Differences between successive angles (turning angles)
            angle_diff = np.diff(angles)  
            # Unwrap to reduce artefactual jumps at ±pi
            angle_diff = np.unwrap(angle_diff)
            angle_var = np.var(angle_diff)  # overall variance
    
            # --- New Temporal Features ---
            # Instantaneous speed trend: regression on disp versus frame (for frames[1:])
            if len(disp) > 1:
                speed_reg = linregress(frames[1:], disp)
                speed_slope = speed_reg.slope  # change in instantaneous speed (nm per frame^2)
            else:
                speed_slope = np.nan
    
            # Turning rate trend: regression on absolute turning angles versus index (0, 1, 2,...)
            if len(angle_diff) > 1:
                turning_reg = linregress(np.arange(len(angle_diff)), np.abs(angle_diff))
                turning_rate_slope = turning_reg.slope  # change in turning (radians per frame)
            else:
                turning_rate_slope = np.nan
    
            # Bounding box area (in nm^2)
            bbox_area = (np.max(x) - np.min(x)) * (np.max(y) - np.min(y))
            
            features.append({
                'cell_id': cell_id,
                'particle': particle,
                'track_duration': duration,
                'mean_speed': speed,
                'net_displacement': net_displacement,
                'directionality_ratio': directionality_ratio,
                'radial_slope': slope,
                'radial_r_squared': r_value**2,
                'radial_change': dists_to_center[-1] - dists_to_center[0],
                'angle_variance': angle_var,
                'motion_bbox_area': bbox_area,
                # New features capturing temporal trends:
                'speed_slope': speed_slope,
                'turning_rate_slope': turning_rate_slope,
            })
        
        return pd.DataFrame(features)
    def _save_centroid_videos_per_cell(self, clusters_binary, all_props, sep_cells, ch='ch0', square_size=3, output_dir='./', filtered_spots=None):
        ch_idx = 0 if ch == 'ch0' else 1
        if self.folder is not None:
            output_dir = os.path.join(self.folder, 'cluster_analysis')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        for cell_idx, cell in enumerate(sep_cells):
            num_frames = cell[ch_idx].shape[0]
    
            orig_stack = []
            bin_stack = []
    
            cell_props = all_props[all_props['cell_id'] == cell_idx]
            if filtered_spots is not None:
                cell_spots = filtered_spots[filtered_spots['cell_id'] == cell_idx]
    
            for frame_num in range(num_frames):
                orig_frame = cell[ch_idx][frame_num]
                binary_frame = clusters_binary[cell_idx][frame_num]
    
                norm = (orig_frame - orig_frame.min()) / (orig_frame.ptp() + 1e-9)
                orig_rgb = (np.dstack([norm] * 3) * 255).astype(np.uint8)
                bin_rgb = np.dstack([binary_frame.astype(np.uint8) * 255] * 3)
    
                binary_uint8 = (binary_frame * 255).astype(np.uint8)
                contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(orig_rgb, contours, -1, (128, 0, 128), 1)
    
                frame_props = cell_props[cell_props['frame'] == frame_num]
                for _, row in frame_props.iterrows():
                    r, c = int(round(row['centroid_row'])), int(round(row['centroid_col']))
                    self._paint_red_square(orig_rgb, (r, c), size=square_size)
                    self._paint_red_square(bin_rgb, (r, c), size=square_size)
    
                # 🟢 Draw filtered spots if provided
                if filtered_spots is not None:
                    frame_spots = cell_spots[cell_spots['t'] == frame_num]
                    for _, spot in frame_spots.iterrows():
                        r, c = int(round(spot['y_per_cell'])), int(round(spot['x_per_cell']))
                        self._paint_red_square(orig_rgb, (r, c), size=square_size, color = [255, 255, 0])
    
                orig_stack.append(orig_rgb)
                bin_stack.append(bin_rgb)
    
            orig_path = os.path.join(output_dir, f"cell{cell_idx}_centroids.tif")
            bin_path = os.path.join(output_dir, f"cell{cell_idx}_binary_centroids.tif")
    
            tifffile.imwrite(orig_path, np.array(orig_stack), photometric='rgb')
            tifffile.imwrite(bin_path, np.array(bin_stack), photometric='rgb')
            print(f"✅ Saved Cell {cell_idx} to:\n- {orig_path}\n- {bin_path}")


    
    # def _save_centroid_videos_per_cell(self, clusters_binary, all_props, sep_cells, ch='ch0', square_size=3, output_dir='./'):
    #     ch_idx = 0 if ch == 'ch0' else 1
    #     if self.folder is not None:
    #         output_dir = os.path.join(self.folder, 'cluster_analysis')
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)

    #     for cell_idx, cell in enumerate(sep_cells):
    #         num_frames = cell[ch_idx].shape[0]

    #         orig_stack = []
    #         bin_stack = []

    #         # Filter dataframe for this cell
    #         cell_props = all_props[all_props['cell_id'] == cell_idx]

    #         for frame_num in range(num_frames):
    #             orig_frame = cell[ch_idx][frame_num]
    #             binary_frame = clusters_binary[cell_idx][frame_num]

    #             # Normalize original image to 0–255
    #             norm = (orig_frame - orig_frame.min()) / (orig_frame.ptp() + 1e-9)
    #             orig_rgb = (np.dstack([norm] * 3) * 255).astype(np.uint8)
    #             bin_rgb = np.dstack([binary_frame.astype(np.uint8) * 255] * 3)

    #             # Draw contours using OpenCV
    #             binary_uint8 = (binary_frame * 255).astype(np.uint8)
    #             contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #             cv2.drawContours(orig_rgb, contours, -1, (128, 0, 128), 1)  # Green on original
    #             # cv2.drawContours(bin_rgb, contours, -1, (0, 255, 0), 1)   # Green on binary

    #             # Filter centroids for this frame
    #             frame_props = cell_props[cell_props['frame'] == frame_num]

    #             for _, row in frame_props.iterrows():
    #                 r, c = int(round(row['centroid_row'])), int(round(row['centroid_col']))
    #                 self._paint_red_square(orig_rgb, (r, c), size=square_size)
    #                 self._paint_red_square(bin_rgb, (r, c), size=square_size)

    #             orig_stack.append(orig_rgb)
    #             bin_stack.append(bin_rgb)

    #         # Save TIFFs
    #         orig_path = os.path.join(output_dir, f"cell{cell_idx}_centroids.tif")
    #         bin_path = os.path.join(output_dir, f"cell{cell_idx}_binary_centroids.tif")

    #         tifffile.imwrite(orig_path, np.array(orig_stack), photometric='rgb')
    #         tifffile.imwrite(bin_path, np.array(bin_stack), photometric='rgb')
    #         print(f"✅ Saved Cell {cell_idx} to:\n- {orig_path}\n- {bin_path}")
   
    def _paint_red_square(self, image, center, size=1, color=None):
        """Paint a red square centered at (row, col) in the RGB image."""
        if color is None:
            color = [255, 0, 0]  # red

        r, c = center
        half = size // 2
        r_start = max(r - half, 0)
        r_end = min(r + half + 1, image.shape[0])
        c_start = max(c - half, 0)
        c_end = min(c + half + 1, image.shape[1])
        image[r_start:r_end, c_start:c_end] = color        
    def _create_mask(self, image_shape, contour):
        mask = np.zeros(image_shape, dtype=bool)
        rr, cc = polygon(contour[:, 1], contour[:, 0], image_shape)
        mask[rr, cc] = True
        return mask
    def _detect_splits_and_merges(self, linked_df, distance_threshold=20, frame_gap=1):
        linked_df = linked_df.copy()
        linked_df['split_event'] = False
        linked_df['merge_event'] = False

        frames = sorted(linked_df['frame'].unique())

        # Build contours dict: {frame: {particle: contour_points}}
        polys_by_frame = {}

        for frame in frames:
            polys_by_frame[frame] = {}
            frame_data = linked_df[linked_df['frame'] == frame]

            for _, row in frame_data.iterrows():
                particle = row['particle']
                contour_points = self.contours_by_frame_and_particle.get(frame, {}).get(particle, None)
                if contour_points is None or len(contour_points) < 3:
                    continue
                polys_by_frame[frame][particle] = contour_points

        for t in frames[:-frame_gap]:
            current_contours = polys_by_frame.get(t, {})
            next_contours = polys_by_frame.get(t + frame_gap, {})

            if not current_contours or not next_contours:
                continue

            # Detect merges: many → one
            for next_part, next_cnt in next_contours.items():
                close_parents = []
                for cur_part, cur_cnt in current_contours.items():
                    dist = self._contour_min_distance(cur_cnt, next_cnt)
                    if dist < distance_threshold:
                        close_parents.append(cur_part)
                if len(close_parents) > 1:
                    linked_df.loc[
                        (linked_df['frame'] == t + frame_gap) & (linked_df['particle'] == next_part),
                        'merge_event'
                    ] = True

            # Detect splits: one → many
            for cur_part, cur_cnt in current_contours.items():
                close_children = []
                for next_part, next_cnt in next_contours.items():
                    dist = self._contour_min_distance(cur_cnt, next_cnt)
                    if dist < distance_threshold:
                        close_children.append(next_part)
                if len(close_children) > 1:
                    linked_df.loc[
                        (linked_df['frame'] == t) & (linked_df['particle'] == cur_part),
                        'split_event'
                    ] = True

        return linked_df
    def _contour_min_distance(self, cnt1, cnt2):
        """
        Compute the minimum Euclidean distance between two contours (Nx2 numpy arrays).
        """
        # cnt1 and cnt2 are arrays of shape (N_points, 2)
        # Compute all pairwise distances and find the minimum
        dists = np.sqrt(np.sum((cnt1[:, None, :] - cnt2[None, :, :])**2, axis=2))
        return np.min(dists)
    def _get_contour_for_particle(self, cell_id, frame, particle):
        """
        Retrieve stored contour points for a given cell/frame/particle.
        Returns None if not found.
        """
        return self.cluster_contours.get(cell_id, {}).get(frame, {}).get(particle, None)
    def _detect_merges_proximity_based(self, linked_df, proximity_threshold=15, frame_gap=1):
        """
        Detect merges based on proximity of clusters at time t and disappearance at t+1.
    
        Parameters:
        -----------
        linked_df : pd.DataFrame
            DataFrame from trackpy with particle tracks and frames.
        proximity_threshold : float
            Distance (in pixels) under which two clusters are considered "close".
        frame_gap : int
            How many frames ahead to check for merging.
        
        Returns:
        --------
        pd.DataFrame : updated linked_df with a new 'merge_event_prox' column
        """
        import numpy as np
    
        linked_df = linked_df.copy()
        linked_df['merge_event_prox'] = False
    
        frames = sorted(linked_df['frame'].unique())
    
        # Build contour lookup
        contours_by_frame = self.contours_by_frame_and_particle
    
        for t in frames[:-frame_gap]:
            frame_df = linked_df[linked_df['frame'] == t]
            next_df = linked_df[linked_df['frame'] == t + frame_gap]
    
            current_particles = frame_df['particle'].values
            next_particles = next_df['particle'].values
    
            for i, row1 in frame_df.iterrows():
                p1 = row1['particle']
                cnt1 = contours_by_frame.get(t, {}).get(p1, None)
                if cnt1 is None or len(cnt1) < 3:
                    continue
    
                for j, row2 in frame_df.iterrows():
                    p2 = row2['particle']
                    if p1 >= p2:
                        continue  # avoid duplicate pairs
    
                    cnt2 = contours_by_frame.get(t, {}).get(p2, None)
                    if cnt2 is None or len(cnt2) < 3:
                        continue
    
                    # Check if these two are close enough to consider for merging
                    dist = self._contour_min_distance(np.array(cnt1), np.array(cnt2))
                    if dist < proximity_threshold:
                        # Now check if only one object exists nearby in next frame
                        possible_merge_target = []
                        for _, row_next in next_df.iterrows():
                            p_next = row_next['particle']
                            cnt_next = contours_by_frame.get(t + frame_gap, {}).get(p_next, None)
                            if cnt_next is None or len(cnt_next) < 3:
                                continue
    
                            d1 = self._contour_min_distance(np.array(cnt1), np.array(cnt_next))
                            d2 = self._contour_min_distance(np.array(cnt2), np.array(cnt_next))
    
                            if d1 < proximity_threshold and d2 < proximity_threshold:
                                possible_merge_target.append(p_next)
    
                        # If exactly one such object exists in the next frame, mark both as merging
                        if len(possible_merge_target) == 1:
                            linked_df.loc[(linked_df['frame'] == t) & (linked_df['particle'].isin([p1, p2])), 'merge_event_prox'] = True
    
        return linked_df

class Single_tracked_folder:
    def __init__(self, folder, ch0_hint = None, ch1_hint = None):
        self.folder = folder
        self.ch0_hint = ch0_hint
        self.ch1_hint = ch1_hint
    def open_files(self):
        yaml_file = self._openyaml()
        if yaml_file:
            ch0_laser = yaml_file['ch0']
            ch1_laser = yaml_file['ch1']
            # print(f'Data has been tracked with the colocalizing tracks algorithm and the reference channel was {ch0_laser}')
            ch0 = load_movie(glob(self.folder + f'/**/**{ch0_laser}nm.tif', recursive=True)[0])[0]
            tracks0 = pd.read_csv(glob(self.folder + f'/**/**{ch0_laser}**_locs_nm_trackpy.csv', recursive=True)[0])
            stats0 =  pd.read_hdf(glob(self.folder + f'/**/**{ch0_laser}**_locs_nm_trackpy_stats.hdf', recursive=True)[0])
            ch1 = load_movie(glob(self.folder + f'/**/**{ch1_laser}nm.tif', recursive=True)[0])[0]
            tracks1 = pd.read_csv(glob(self.folder + f'/**/**{ch1_laser}**_locs_nm_trackpy.csv', recursive=True)[0])
            stats1 = pd.read_hdf(glob(self.folder + f'/**/**{ch1_laser}**_locs_nm_trackpy_stats.hdf', recursive=True)[0])
            coloc_tracks = pd.read_csv(glob(self.folder + '/**/**colocsTracks.csv', recursive=True)[0])
            coloc_stats = pd.read_hdf(glob(self.folder + '/**/**colocsTracks_stats.hdf', recursive=True)[0])
            image1 = Tracked_image(
                                   ch0,
                                   tracks0, 
                                   stats0,
                                   ch1, 
                                   tracks1, 
                                   stats1,
                                   coloc_tracks,
                                   coloc_stats, 
                                   self._get_px2nm(),   
                                   self.folder
                                   )
            return image1
        elif not yaml_file and len(glob(self.folder + '/**/**_locs_nm_trackpy.csv', recursive=True)) == 2:
            expected_columns = [
                "track.id", "cell_id", "x_std", "y_std", "length", "loc_count",
                "msd_fit_a", "msd_fit_b", "jumps", "lagtimes", "msd",
                "D_jd0", "A_jd0", "D_jd1", "A_jd1", "D_msd",
                "path", "contour", "area", "centroid"
            ]
            ch0_laser = self.ch0_hint
            ch1_laser = self.ch1_hint
            # print(f'Data has been tracked with the colocalizing tracks algorithm and the reference channel was {ch0_laser}')
            ch0 = load_movie(glob(self.folder + f'/**/**{ch0_laser}nm.tif', recursive=True)[0])[0]
            tracks0 = pd.read_csv(glob(self.folder + f'/**/**{ch0_laser}**_locs_nm_trackpy.csv', recursive=True)[0])
            stats0 = next((pd.read_hdf(f) for f in glob(self.folder + f'/**/**{ch0_laser}**_locs_nm_trackpy_stats.hdf', recursive=True) if os.path.isfile(f)), pd.DataFrame(columns = expected_columns))
            ch1 = load_movie(glob(self.folder + f'/**/**{ch1_laser}nm.tif', recursive=True)[0])[0]
            tracks1 = pd.read_csv(glob(self.folder + f'/**/**{ch1_laser}**_locs_nm_trackpy.csv', recursive=True)[0])
            stats1 = next((pd.read_hdf(f) for f in glob(self.folder + f'/**/**{ch1_laser}**_locs_nm_trackpy_stats.hdf', recursive=True) if os.path.isfile(f)), pd.DataFrame(columns = expected_columns))
            image1 = Tracked_image(
                                   ch0,
                                   tracks0, 
                                   stats0,
                                   ch1, 
                                   tracks1, 
                                   stats1, 
                                   px2nm = self._get_px2nm(), 
                                   folder = self.folder
                                   )
            return image1
        elif not yaml_file and len(glob(self.folder + '/**/**_locs_nm_trackpy.csv', recursive=True)) == 1:
            ch0 = load_movie(glob(self.folder + '/**/**nm.tif', recursive=True)[0])[0]
            tracks0 = pd.read_csv(glob(self.folder + '/**/**locs_nm_trackpy.csv', recursive=True)[0])
            stats0 =  pd.read_hdf(glob(self.folder + '/**/**locs_nm_trackpy_stats.hdf', recursive=True)[0])
            image1 = Tracked_image(
                                   ch0,
                                   tracks0, 
                                   stats0,
                                   px2nm = self._get_px2nm(), 
                                   folder = self.folder
                                   )
            return image1
        else: 
            print('This object is called: "Single_tracked_folder". So, please, at least track something, my friend ;)')          
    def validate(self):
        # ch0, tracks0, stats0, ch1=None, tracks1=None, stats1=None, coloc_tracks=None, coloc_stats=None, px2nm=108
            # print(f'Data has been tracked with the colocalizing tracks algorithm and the reference channel was {ch0_laser}')
        tifs = glob(self.folder + '/**/**nm.tif', recursive=True)
        csvs = glob(self.folder + '/**/**_locs_nm_trackpy.csv', recursive=True)
        coloc_tracks = glob(self.folder + '/**/**_colocsTracks.csv', recursive=True)
        print('The images present are:')
        for i in tifs:
            print(i)
        print('\nThe tracked images are:')
        for i in csvs:
            print(i)
        if coloc_tracks:
            yaml_file = self._openyaml()
            ch0_laser = yaml_file['ch0']
            print(f'\nThe tarcks have been colocalized with {ch0_laser}nm as reference channel')

        else:
            print('The tracks have not been colocalized')   
    def _openyaml(self):
        name = glob(self.folder + '/**/**_colocsTracks.yaml', recursive=True)
        if name:
            with open(name[0], 'r') as file:
                data_parts = list(yaml.safe_load_all(file))
            data = {}
            for i in data_parts:
                data.update(i)
            return data
        else:
            return False
    def _get_px2nm(self): #if self.transform = True, this will get the correct naclib coefficients (Annapurna VS K2)
        resultPath  = glob(self.folder + '/**/**result.txt', recursive=True)[0]
        result_txt  = tools.read_result_file(resultPath) #this opens the results.txt file to check the microscope used. 
                #It should be in a folder called paramfile inside the folder where the script is located. 
        if result_txt['Computer'] == 'ANNAPURNA': 
            return 90.16
        elif result_txt['Computer'] == 'K2-BIVOUAC':
            return 108

class Dataset_tracked_folder:
    def __init__(self, folder):
        self.folder = folder
        self.conditions = self._get_conditions()
        self._conditions_paths = [os.path.join(self.folder, subfolder) for subfolder in self.conditions]
        self.conditions_to_use = self.conditions
        self.tracked_folders = dict.fromkeys(self.conditions_to_use, 'Not analyzed, use count_tracked()')
        self.result_count = None
    def _get_conditions(self):
        subfolders = [f.name for f in os.scandir(self.folder) if f.is_dir()]
        return subfolders
    def select_conditions(self, indices):
        self.conditions_to_use = [self.conditions[i] for i in indices]
        self._conditions_paths = [os.path.join(self.folder, subfolder) for subfolder in self.conditions_to_use]
    def count_cotracked(self):
        for i in range(len(self._conditions_paths)):
            pathshdf = glob(self._conditions_paths[i] + '/**/**colocsTracks_stats.hdf', recursive=True)
            pathshdf = [os.path.dirname(path) for path in pathshdf]
            num_runs = self._count_run_folders_recursive(self._conditions_paths[i])
            print(f'from {self._conditions_paths[i]}, {len(pathshdf)} runs have colocalized tracks out of {num_runs} runs')
            self.tracked_folders[self.conditions_to_use[i]] = pathshdf
    def count_number_cotracks(self):
        results = []  # This will store the data for each run folder
        run_pattern = re.compile(r'^Run\d+$')  # Regex to identify folders like 'Run0001'
        path_yaml = glob(self.folder + r'/**/*_colocsTracks.yaml', recursive=True)
        general_yaml_file = self._openyaml(path_yaml)
        ch0_hint = general_yaml_file['ch0']
        ch1_hint = general_yaml_file['ch1']
        min_len_track = general_yaml_file["min_len_track"]
        # Loop through each selected condition and its corresponding path
        for cond_path, cond_name in zip(self._conditions_paths, self.conditions_to_use):
            # Recursively walk through the directory tree under each condition
            for dirpath, dirnames, _ in os.walk(cond_path):
                for dirname in dirnames:
                    if run_pattern.match(dirname):  # Check if the folder matches 'RunXXXX'
                        run_folder = os.path.join(dirpath, dirname)
                        # Try to find a YAML file in the run folder (used to get min_len_track)
                        yaml_files = glob(run_folder + '/**/*_colocsTracks.yaml', recursive=True)
                        if yaml_files:
                            yaml_data = self._openyaml(yaml_files)
                            min_len_track = yaml_data.get("min_len_track", 5)  # Default to 5 if key missing

                        # Load tracking data using 
                        a = Single_tracked_folder(run_folder, ch0_hint, ch1_hint).open_files()
                        print(run_folder)
                        # Count tracks in channel 0 that meet the length threshold
                        if isinstance(a.stats0, pd.DataFrame):
                            ch0_tracks = a.stats0[a.stats0.loc_count >= min_len_track].shape[0]
                        else: 
                            ch0_tracks = 0
                        if isinstance(a.stats1, pd.DataFrame):
                            ch1_tracks = a.stats1[a.stats1.loc_count >= min_len_track].shape[0]
                        else: 
                            ch1_tracks = 0
                        try:
                            # Count colocalized tracks if the attribute exists
                            coloc_count = a.coloc_stats.shape[0] if hasattr(a, 'coloc_stats') else 0
                            
                        except Exception as e:
                            # If loading fails, assume 0 tracks
                            coloc_count = 0

                        # Append the results for this run folder
                        results.append({
                            "folder": run_folder,
                            "condition": cond_name,
                            "colocalized_tracks": coloc_count,
                            "ch0_tracks": ch0_tracks,
                            "ch1_tracks": ch1_tracks,
                            "min_len_track": min_len_track
                        })

        # Convert the list of dictionaries into a pandas DataFrame
        self.result_count = pd.DataFrame(results)
        return self.result_count
    def summary_count_number_cotracks(self):
        if isinstance(self.result_count, pd.DataFrame):
            aggregated_df = self.result_count.groupby('condition').agg(
                total_colocalized_tracks=pd.NamedAgg(column='colocalized_tracks', aggfunc='sum'),
                total_ch0_tracks=pd.NamedAgg(column='ch0_tracks', aggfunc='sum'),
                total_ch1_tracks=pd.NamedAgg(column='ch1_tracks', aggfunc='sum')
            ).reset_index()
            return aggregated_df
        else: 
            print("run count_number_cotracks first")
    def get_Ds(self, min_len = 10, channel = 'ch0'):
        all_ds = []  # List to collect all DataFrames
        box = BoxPlotter(xlabel="Conditions", ylabel="Diff. Coeff. (um^2/sec)")
        for i, cond in tqdm(zip(self._conditions_paths, self.conditions_to_use), desc='Extracting Ds...\n'):
            print(f'\nAnalyzing {i}...')
            ds_cond = []
            pathshdf = glob(i + '/**/**.hdf', recursive=True)
            paths_locs = list(set(os.path.dirname(file) for file in pathshdf))
            for j in paths_locs:
                image = Single_tracked_folder(j).open_files()
                ds = image.extract_Ds(min_len, channel)
                if isinstance(ds, pd.DataFrame):
                # Add columns at the beginning
                    ds.insert(0, 'condition', cond)
                    ds.insert(0, 'run', j)
                    all_ds.append(ds)  # Accumulate
                    ds_cond.append(ds)
            final_cond = pd.concat(ds_cond, ignore_index=True)
            box.add_box(final_cond.D_msd, cond)
        # Concatenate all DataFrames into one
        final_ds = pd.concat(all_ds, ignore_index=True)
        # box.add_statistical_annotations()
        return final_ds, box
    def get_dwell(self, min_len = 10, ref = 'ch0', x0=0, xt=None, y0=0, yt=None):
        all_dwell = []
        frame_rate = self._get_frame_rate()
        hist = HistogramPlotter( xlabel="dwell_time(sec)", ylabel="Frequency")
        for i, cond in tqdm(zip(self._conditions_paths, self.conditions_to_use), desc='Extracting Ds...\n'):
            print(f'\nAnalyzing {i}...')
            dwell_cond = []
            pathshdf = glob(i + '/**/**colocsTracks_stats.hdf', recursive=True)
            paths_locs = list(set(os.path.dirname(file) for file in pathshdf))
            for j in tqdm(paths_locs):
                pathsyaml = glob(j + '/**/**colocsTracks.yaml', recursive=True)
                yaml = self._openyaml(pathsyaml)
                image = Single_tracked_folder(j).open_files()
                dwell = image.extract_dwell(frame_rate = frame_rate, min_len = min_len, max_dist = yaml['th'], ref = ref)
                if isinstance(dwell, pd.DataFrame):
                # # Add columns at the beginning
                    dwell.insert(0, 'condition', cond)
                    dwell.insert(0, 'run', j)
                    all_dwell.append(dwell)  # Accumulate
                    dwell_cond.append(dwell)
            final_cond = pd.concat(dwell_cond, ignore_index=True)
            hist.add_data(final_cond.dwell_time, label = f"{cond}")
        final_dwell = pd.concat(all_dwell, ignore_index=True)
        hist.set_labels()
        hist.set_xlim(x0, xt)
        hist.set_ylim(y0, yt)
        hist.show_plot()
        # box.add_statistical_annotations()
        return final_dwell, hist
    def validate(self):
        print("Just a reminder that most of the time the whole dataset should be analyzed using the same parameters.")
        print("Here are the parameters for the first folder in the dataset that has colocalized tracks:")
        yaml = self._openyaml(glob(self.folder + r'/**/*_colocsTracks.yaml', recursive=True))
        for i, j in enumerate(yaml.items()):
            print(f"{j[0]}: {j[1]}")
    def _get_px2nm(self): #if self.transform = True, this will get the correct naclib coefficients (Annapurna VS K2)
        resultPath  = glob(self.folder + '/**/**result.txt', recursive=True)[0]
        result_txt  = tools.read_result_file(resultPath) #this opens the results.txt file to check the microscope used. 
                #It should be in a folder called paramfile inside the folder where the script is located. 
        if result_txt['Computer'] == 'ANNAPURNA': 
            return 90.16
        elif result_txt['Computer'] == 'K2-BIVOUAC':
            return 108
    def _get_frame_rate(self): #if self.transform = True, this will get the correct naclib coefficients (Annapurna VS K2)
        resultPath  = glob(self.folder + '/**/**result.txt', recursive=True)[0]
        result_txt  = tools.read_result_file(resultPath) #this opens the results.txt file to check the microscope used. 
                #It should be in a folder called paramfile inside the folder where the script is located. 
        time_unit = result_txt['Interval']
        time = time_unit.split(' ')[0]
        unit = time_unit.split(' ')[1]
        if unit == 'sec':
            return float(time)
        elif unit == 'ms':
            return int(time)/1000
        elif unit == 'min':
            return int(time)*1000
           
    def _count_run_folders_recursive(self, root_folder):
        pattern = re.compile(r'^Run\d+$')
        count = 0

        for dirpath, dirnames, _ in os.walk(root_folder):
            for dirname in dirnames:
                if pattern.match(dirname):
                    count += 1
        return count
    def _openyaml(self, name):
        # Load YAML content if found
        with open(name[0], 'r') as file:
            data_parts = list(yaml.safe_load_all(file))
        yaml_data = {}
        for part in data_parts:
            yaml_data.update(part)
        return yaml_data