# -*- coding: utf-8 -*-
"""Data Reader module

This module is used to the data I/O. The main uses are to convert root files to parquet files (requires a installation
of ROOT with pyROOT enabled) and to later load them for the data processing.


Example
-------
You can read root files using the read_root_file function. The data_reader is imported as dr.

    $ import data_reader as dr

First you can se the location which the files will be saved with the set_storage_location function. If no path is set,
the data is saved under a folder data in the current folder:

    $ dr.set_storage_location("path_to_folder_which_the_data_will_be_saved")

Then you should configure the base name of the folders. By default, they are set to ``DHFeCorrelation_`` which is the
one used in the C++/ROOT interface. You can change it if necessary with set_base_folder_name:

    $ dr.set_storage_location("name_of_the_base_folder_in_the_ROOT_file")

You can finally read the file using the read_root_file by setting the file name and the configuration name. This should
be the value that follows the base name in the ROOT file: for example in "DHFeCorrelation_loose",
configuration_name is ``loose``.

    $ electrons, d_mesons = dr.read_root_file("AnalysisResults.root", "loose")

Than the data can be saved in the parquet file using

    $ dr.save(electrons, "loose", "electron", 265534)

and later loaded using:
    electron = dr.load("loose", "electron", 265534)


"""
import glob
import itertools
import os
import random
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

import dhfcorr.definitions as definitions


def read_root_file(file_name, configuration_name, particles=('electron', 'dmeson'), base_folder_name='DHFeCorrelation_',
                   backend='uproot', return_dict=False, **kwargs):
    """Read the root file with name file_name (should include the path to the file) and configuration named
    configuration_name. Returns DataFrame with the contents of the associated ROOT.TTree.

    Parameters
    ----------
    file_name : str
        Name of the file that contains the ROOT TTree. Should contain the path to the file.

    configuration_name: str
        The name of the configuration. It can be obtained by checking the name of the folder inside file_name. It should
        start with the value set to base_folder_name. It is a parameter in the AddTask.

    particles: tuple
        name of the trees that contain the particles

    base_folder_name: str
        name of the folder

    backend: str
        Package used to convert root to pandas DataFrames. 'uproot' does not require a ROOT instalation, while
        'root_pandas' requires it.

    **kwargs:
        parameters to be forwarded to root_pandas.read_root

    Returns
    -------
    data_frames : list[pd.DataFrame] or pd.DataFrame
        DataFrame with the data of each tree. If only one particle name was passed, a single dataframe is returned

    """
    folder_name = base_folder_name + configuration_name

    if isinstance(particles, str):
        particles = [particles]

    if backend == 'uproot':
        try:
            import uproot
        except ModuleNotFoundError:
            raise ModuleNotFoundError("No uproot found and backend='uproot', please install it. ")

        file = uproot.open(file_name)
        data_frames = [file.get(folder_name + '/' + x).pandas.df(flatten=False) for x in particles]

    elif backend == 'root_pandas':
        try:
            from root_pandas import read_root
        except ModuleNotFoundError:
            raise ModuleNotFoundError("No root_pandas found and backend='root_pandas', please install it. ")
        data_frames = [read_root(file_name, folder_name + '/' + x, **kwargs) for x in particles]

    else:
        raise ValueError('The backend passed is invalid. Please use uproot or root_pandas.')

    if return_dict:
        return {x: df for x, df in zip(particles, data_frames)}

    if len(data_frames) == 1:
        data_frames = data_frames[0]

    return data_frames


def save(df, dataset_name, particle=None, run_number=None, step='raw', index=False, file_name=None):
    """Saves the dataset into a parquet file in the default storage location. The directory is created in case it does
    not exist.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame that will be saved.

    dataset_name: str
        The name of the configuration. It can be obtained by checking the name of the folder inside file_name. It should
        start with the value set to base_folder_name. It is a parameter in the AddTask.

    particle: str
        The particle name, such as ``electron` or ``dmeson``. The same name will have to be used to load it.

    run_number: str or possible to convert to str
        This is a unique identifier for each file. Usually the run number or period is used.

    step: str
        Name of the step. The files will be saved to 'processing folder + step'

    index: bool
        Set to include/exclude the index in the file saved.

    """

    location = get_location_step(dataset_name, step)
    check_for_folder(location)
    check_for_folder(definitions.PROCESSING_FOLDER + dataset_name)

    if file_name is None:
        if run_number is None or particle is None:
            raise ValueError('You should provide at file_name or run_number and particle')
        name_to_save = location + "/" + str(run_number) + '_' + particle + '.parquet'
    else:
        name_to_save = location + file_name

    df.to_parquet(name_to_save, index=index)


class LazyFileLoader:
    def __init__(self, file='', index=None, columns=None):
        self.file_name = file
        self.columns = columns
        self.index = index

    def __copy__(self):
        return LazyFileLoader(self.file_name, self.index, self.columns)

    def __str__(self):
        return self.file_name

    def __eq__(self, other):
        return self.file_name == other.file_name

    def load(self, columns=None, index=None):

        if columns is None:
            columns = self.columns
        if index is None:
            index = self.index

        if index is not None:
            df = pd.read_parquet(self.file_name, columns=columns).set_index(index)
        else:
            df = pd.read_parquet(self.file_name, columns=columns)

        reduce_dataframe_memory(df)

        return df


def get_files_from_runlist(configuration_name, run_numbers, particle, step='raw'):
    location = get_location_step(configuration_name, step)
    print(location)
    print(particle)
    files = [glob.glob(location + "*" + str(run) + '*_*' + str(particle) + '*.parquet') for run in run_numbers]
    files = list(itertools.chain.from_iterable(files))

    return files


def find_missing_processed_files(dataset_name, input_step, output_step, particle, run=None,
                                 full_file_path=False):
    """Find the files that were not yet processed at the output_step.

    Parameters
    ------
    dataset_name: str
       Name of the dataset in the local file system.
    input_step: str
       The input step that will be processed.
    output_step: str
        The output step that will be processed.
    particle: str or None
        The particle (or other form of identifier, e.g. event) that will be processed.
    run: list
        In case it is not None, this will search only for files with this run/period names
    full_file_path: bool
        By default, the names of the files are returned. In case you need the full path in the file system, pass True.

    Returns
    ------
    missing_files: list
        The name of the files which have not yet been processed. If full_file_path is true, the whole path is returned.
    """

    files_input = {get_friendly_name(x) for x in get_file_list(dataset_name, particle, input_step, run)}
    if len(files_input) == 0:
        raise ValueError('The input step has no files.')
    files_output = {get_friendly_name(x) for x in get_file_list(dataset_name, particle, output_step, run)}

    missing_files = list(files_input - files_output)

    if full_file_path:
        missing_files = [find_file(dataset_name, x, input_step) for x in missing_files]

    return list(missing_files)


def find_file(config, name, step):
    path = get_location_step(config, step)
    return glob.glob(path + '*' + name + '*')[0]


def search_for_processed(dataset_name, input_step, output_step, particle='dmeson'):
    files = find_missing_processed_files(dataset_name, input_step, output_step, particle)
    runs_missing = {get_run_number(x) for x in files}
    return list(runs_missing)


def get_location_step(dataset_name, step='raw'):
    """Returns the location of the files of a given step.

     Parameters
     ------
     dataset_name: str
        Name of the dataset in the local file system.
     step: str
        The step of the analysis.

    Returns
    ------
    str:
        the location of the step
     """

    if step == 'ml' or step == 'training' or step == 'test' or step == 'ml-dataset':
        return definitions.PROCESSING_FOLDER + dataset_name + '/ml-dataset/'

    if step == 'root_merged' or step == 'root':
        return definitions.DATA_FOLDER + '/' + step + '/' + dataset_name + '/'

    return definitions.PROCESSING_FOLDER + dataset_name + '/' + step + '/'


def get_ml_dataset(dataset_name, yaml_config, pt_bin, step='train', extra_cols=None):
    if extra_cols is None:
        extra_cols = yaml_config.values['model_building']['extra_cols']

    features = yaml_config.values['model_building']['features']
    target = yaml_config.values['model_building']['target']
    location = get_location_step(dataset_name, 'ml')

    file = location + 'ml_sample_' + step + '_' + str(pt_bin) + '.parquet'
    df = pd.read_parquet(file, columns=features + [target] + extra_cols)

    df[target] = df[target] > -1

    return df


def get_cv_results(dataset_name, prefix=''):
    files = glob.glob(get_location_step(dataset_name, 'ml') + prefix + 'cv*.')
    files = sorted(files, key=lambda name: int(name.split('/')[-1].split('_')[-1].split('.')[0]))
    dfs = [pd.read_csv(x) for x in files]
    return dfs


def get_periods(configuration, step='raw'):
    location = get_location_step(configuration, step)
    file_list = glob.glob(location + "/*.parquet")
    periods_from_files = list({get_period(x) for x in file_list})
    return periods_from_files


def get_run_numbers(configuration, step='raw'):
    location = get_location_step(configuration, step)
    file_list = glob.glob(location + "/*.parquet")
    runs_from_files = list({get_run_number(x) for x in file_list})

    return runs_from_files


def get_file_list(configuration_name, particle=None, step='raw', run=None):
    search_path = get_location_step(configuration_name, step)

    if run is not None:
        search_path += '*' + str(run) + "*"
    if particle is not None:
        search_path += particle
    search_path += '*'

    return glob.glob(search_path)


def find_models(dataset, compiled=False, prefix=''):
    """"Find models for dataset. In case more than one configuration was used, you should use the prefix to specify it.

    Parameters
    ---------
    dataset: str
        name of the dataset in the file system
    prefix: str
        additional value used as a prefix to save different models in the same dataset
    compiled: bool
        if True, looks for compiled treelite models

    Returns
    ----
    files: list
        list of the location of the files. They are ordered by the index following 'model_', usually the pT.

    """

    extension = 'txt'
    if compiled:
        extension = 'so'
    files = glob.glob(get_location_step(dataset, 'ml') + prefix + 'model*.' + extension)
    files = sorted(files, key=lambda name: int(name.split('/')[-1].split('_')[-1].split('.')[0]))

    return list(files)


def load(dataset_name, particle,
         step='raw',
         run_number=None, columns=None,
         index=None, sample_factor=None,
         lazy=False):
    """Loads the dataset from the default storage location. If run_number is a list, all the runs in the list will be
    merged.

    Parameters
    ----------
    dataset_name: str
        Name of the dataset in the local file system

    particle: str or list
        The particle name, such as ``electron` or ``dmeson``. The same name that was used to save it.

    step: str
        Step of the analysis.

    run_number: str, list or None
        This is a unique identifier for each file. Usually the run number is used.
        If None, the function will read all the files, unless lazy=True.

    columns: list:
        If not None, only these columns will be read from the file.

    index: list or None:
        The resulting dataframes will be indexed using the provided values. If None, the index is not meaningful.

    sample_factor: None or float:
        Loads only part of the files, if a number between 0 and 1 is provided.

    lazy: bool
        In case run_number = None, lazy=True will not load all the files, but rather return a LazyFileLoader which needs
        to be called with the load method to lead the data.

    Warnings
    ----------
    UserWarning
        In case one of the run data is not loaded.

    Raises
    ----------
    ValueError
        If no data is loaded.

    """
    if isinstance(particle, (str, int, float)):
        particle = [particle]

    if isinstance(run_number, (str, int, float)):
        run_number = [run_number]

    if columns is not None:
        if len(particle) == 1 and len(columns) != 1:
            columns = [columns]

    for x in index:
        for col_part in columns:
            if x not in col_part:
                col_part.append(x)

    location = get_location_step(dataset_name, step)

    # Add index in case it is not in columns
    print(columns)

    if run_number is None:
        file_list = glob.glob(location + "/*" + str(particle[0]) + ".parquet")
        run_list = [get_friendly_parquet_file_name(x, particle[0]) for x in file_list]
        run_list.sort()
        return load(dataset_name, particle, step, run_list, columns, index, sample_factor, lazy)
    else:
        file_list = [[location + r"/" + str(run) + '_' + str(x) + '.parquet' for x in particle] for run in run_number]

    if sample_factor is not None:
        if sample_factor > 1.:
            raise ValueError("It is not possible to sample for than 1")
        number_to_sample = int(sample_factor * len(file_list))
        print("Sampling only " + str(number_to_sample) + ' out of ' + str(len(file_list)) + ' files')
        file_list = random.sample(file_list, number_to_sample)

    if lazy:
        lazy_list = [[LazyFileLoader(x, index=index, columns=col) for x, col in zip(list_run, columns)]
                     for list_run in file_list]
        # Reduce dimensionality in case it is only one particle
        if len(particle) == 1:
            lazy_list = [x[0] for x in lazy_list]

        return lazy_list

    data_sets = list()
    for f in tqdm(file_list):
        for file_particle, col in zip(f, columns):
            try:
                df = pd.read_parquet(file_particle, columns=col)
                data_sets.append(df)
            except OSError:
                warnings.warn('It is not possible to load the file ' + str(f))
                return None

    if len(data_sets) < 0:
        raise ValueError('No data was loaded.')

    result_dataset = pd.concat(data_sets, sort=False)

    if index is not None:
        result_dataset.set_index(index, inplace=True)

    reduce_dataframe_memory(result_dataset)

    return result_dataset


def get_friendly_root_file_name(file):
    return file.split('/')[-1][:-5]


def get_friendly_parquet_file_name(file, particle=''):
    return file.split('/')[-1].split('.')[0][:-(len(particle) + 1)]


def get_dataset_name_from_file(file, step='raw'):
    if step == 'root_merged' or step == 'root':
        return file.split(definitions.DATA_FOLDER + '/' + step + '/')[-1].split('/')[0]
    return file.split(definitions.PROCESSING_FOLDER)[-1].split('/')[0]


def get_particle_from_file(file):
    return file.split('/')[-1].split('.')[0].split('_')[-1]


def get_file_name(x):
    return x.split('/')[-1]


def get_friendly_name(x):
    if '.root' in x:
        return x.split('/')[-1][:-5]
    else:
        return ''.join([a + '_' for a in x.split('/')[-1].split('.')[0].split('_')[:-1]])[:-1]


def reduce_dataframe_memory(df):
    for col in df.columns[df.dtypes == 'float64']:
        df[col] = df[col].astype(np.float32)

    for col in df.columns[df.dtypes == 'int64']:
        df[col] = df[col].astype(np.int32)

    return df


def get_period(location):
    return location.split('/')[-1].split('_')[0]


def get_run_number(location):
    return int(location.split('/')[-1].split('_')[1])


def split_files(file, size, max_file_size):
    if int(len(file)) != int(len(size)):
        raise ValueError('Arrays with file names and file sizes do not have the same length.\n'
                         'Lengths: files ' + str(len(file)) + ' and size ' + str(len(size)))
    if int(max_file_size) <= 0:
        raise ValueError('Maximum size should be > 0')

    files_info = pd.DataFrame({'file': file, 'size': size}).set_index('file')
    files_left = files_info.copy()

    large_files = files_left[files_left['size'] >= float(max_file_size)]

    if len(large_files) > 0:  # Add all files that have size > max size
        groups = [[x] for x in large_files.index.values]
        files_left.drop(large_files.index, inplace=True)
    else:  # Create a group only with the first file
        groups = list()
        groups.append([files_left.index[0]])
        files_left.drop(files_left.index[0], inplace=True)

    while len(files_left) > 0:
        current_group = groups[-1]

        # Check which files are still allowed in the current group
        size_left = max_file_size - (files_info.loc[current_group]['size'].sum())
        files_that_fit = files_left.loc[files_info['size'] <= size_left].index

        if len(files_that_fit) > 0:  # Still possible to add more files
            current_group.append(files_that_fit[0])
            files_left.drop(files_that_fit[0], inplace=True)
            continue
        else:  # Current group is full, create a new group
            new_group = [files_left.index[0]]
            files_left.drop(new_group, inplace=True)
            groups.append(new_group)

    return groups


def check_for_folder(folder):
    if folder is None:
        return
    if not os.path.isdir(folder):
        os.mkdir(folder)
