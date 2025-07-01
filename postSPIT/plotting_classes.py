from picasso.io import TiffMultiMap, load_movie
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from tqdm import tqdm
import os
from glob import glob
import yaml
from spit import tools
import re

class Plotter:
    def __init__(self, title="Plot", xlabel="X-axis", ylabel="Y-axis", figsize = (6.4, 4.8)):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.fig, self.ax = plt.subplots(figsize=figsize)
    def set_labels(self):
        # self.ax.set_title(self.title)
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
    def add_data(self, data, bins=10, label="Histogram", color="purple"):
        self.ax.hist(data, bins=bins, label=label, color=color)

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
    def __init__(self, ch0, tracks0, stats0, ch1=None, tracks1=None, stats1=None, coloc_tracks=None, coloc_stats=None, px2nm=108):
        #checking data types
        assert isinstance(ch0, TiffMultiMap), "'ch0' must be a picasso.io.TiffMultiMap object"
        assert isinstance(tracks0, pd.DataFrame), "'tracks0' must be a pd.DataFrame object"
        assert isinstance(stats0, pd.DataFrame), "'stats0' must be a pd.DataFrame object"
        assert isinstance(px2nm, (int, float))
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
        self.px2nm = px2nm
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
    
        plotter = TrackPlotter(image, self.px2nm)
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
    
        plotter = TrackPlotter(self.ch0, self.px2nm)
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
                track.at[index, 'im_int_0'] = np.mean(self.ch0[row.t, int(row.y_0/self.px2nm)-1:int(row.y_0/self.px2nm)+1, int(row.x_0/self.px2nm)-1:int(row.x_0/self.px2nm)+1]) / np.median(self.ch0[row.t])
            if not np.isnan(row.y_1):
                track.at[index, 'im_int_1'] = np.mean(self.ch1[row.t, int(row.y_1/self.px2nm)-1:int(row.y_1/self.px2nm)+1, int(row.x_1/self.px2nm)-1:int(row.x_1/self.px2nm)+1]) / np.median(self.ch1[row.t])

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
    def extract_Ds(self, min_len = 10, channel = 'ch0'):
        if channel == 'ch0':
            stats = self.stats0
        elif channel == 'ch1':
            assert self.stats1 is not None, "stats1 is not provided."
            stats = self.stats1
        else:
            raise ValueError("channel must be 'ch0' or 'ch1'")
        columns_to_extract = ['track.id', 'cell_id', 'D_msd']
        ds = stats[(stats['length'] >= min_len)][columns_to_extract]
        ds = ds.dropna(subset = ['D_msd'])        
        return ds
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
                                   self._get_px2nm()                     
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
                                   px2nm = self._get_px2nm()                     
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
                                   px2nm = self._get_px2nm()                     
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
    def get_Ds(self):
        for i in tqdm(self._conditions_paths, desc = 'Extracting Ds...\n'):
            print(f'\nAnalyzing {i}...')
            pathshdf = glob(i + '/**/**.hdf', recursive=True)
            paths_locs = list(set(os.path.dirname(file) for file in pathshdf))
            for j in paths_locs:
                image = Single_tracked_folder(j).open_files()
                print(image.extract_Ds())
    def _get_px2nm(self): #if self.transform = True, this will get the correct naclib coefficients (Annapurna VS K2)
        resultPath  = glob(self.folder + '/**/**result.txt', recursive=True)[0]
        result_txt  = tools.read_result_file(resultPath) #this opens the results.txt file to check the microscope used. 
                #It should be in a folder called paramfile inside the folder where the script is located. 
        if result_txt['Computer'] == 'ANNAPURNA': 
            return 90.16
        elif result_txt['Computer'] == 'K2-BIVOUAC':
            return 108
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