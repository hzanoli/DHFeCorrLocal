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

Attributes
----------
storage_location: str
    Location used to save the parquet files.
base_folder_name: str
    Base name used in the folders inside the ROOT File.
tree_name : dict
    The name of the trees for the electrons and D mesons in the ROOT file.


"""

import warnings
import pandas as pd
import numpy as np
import glob
import dhfcorr.config_yaml as configyaml
import dhfcorr.definitions as definitions
import random

storage_location = definitions.DATA_FOLDER
base_folder_name = "DHFeCorrelation_"


def set_storage_location(location):
    """"Set the location that the parquet files will be saved.

    Parameters
    ----------
    location: str
        Location that the parquet files will be saved.
    """
    global storage_location
    storage_location = location


def set_base_folder_name(name):
    global base_folder_name
    base_folder_name = name


def read_root_file(file_name, configuration_name, particles=('electron', 'dmeson'), **kwargs):
    """Read the root file with name file_name (should include the path to the file) and configuration named
    configuration_name. Returns DataFrame with the contents of the associated ROOT.TTree.

    Parameters
    ----------
    file_name : str or list with str
        Name of the file that contains the ROOT TTree. Should contain the path to the file.
        In case multiple files are provided in a list, all of them are loaded. In this case it is recommended to use
        the kwarg chunksize with the number of rows (chunksize=2500000 is recommended for 16GB of RAM)

    configuration_name: str
        The name of the configuration. It can be obtained by checking the name of the folder inside file_name. It should
        start with the value set to base_folder_name. It is a parameter in the AddTask.

    particles: tuple
        name of the trees that contain the particles

    **kwargs:
        parameters to be forwarded to root_pandas.read_root

    Returns
    -------
    data_frames : list(pd.DataFrame)
        DataFrame with the data of each tree

    """
    from root_pandas import read_root

    folder_name = base_folder_name + configuration_name

    data_frames = [read_root(file_name, folder_name + '/' + x, **kwargs) for x in particles]

    if len(data_frames) == 1:
        data_frames = data_frames[0]

    return data_frames


def save(df, configuration_name, particle, run_number):
    """Saves the dataset into a parquet file in the default storage location. The directory is created in case it does
    not exist.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame that will be saved.

    configuration_name: str
        The name of the configuration. It can be obtained by checking the name of the folder inside file_name. It should
        start with the value set to base_folder_name. It is a parameter in the AddTask.

    particle: str
        The particle name, such as ``electron` or ``dmeson``. The same name will have to be used to load it.

    run_number: str, float or int
        This is a unique identifier for each file. Usually the run number is used. It should be

    """
    import os
    if not os.path.isdir(storage_location + configuration_name):
        os.mkdir(storage_location + configuration_name)

    name_to_save = storage_location + configuration_name + r"/" + str(run_number) + '_' + particle + '.parquet'
    df.to_parquet(name_to_save, index=False)


class LazyFileLoader:

    def __init__(self, file='', index=None, columns=None):
        self.file_name = file
        self.columns = columns
        self.index = index

    def __copy__(self):
        return LazyFileLoader(self.file_name, self.index, self.columns)

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


def get_file_name(config, stage):
    file_name = None
    base_folder = config.values['base_folder']
    if stage == 'raw':
        file_name = config.values['pair_file']
    elif stage == 'selected':
        file_name = config.values['selected_pair_file']

    return file_name


def load_pairs(config_file, stage='raw'):
    if isinstance(config_file, configyaml.ConfigYaml):
        config = config_file
    else:
        config = configyaml.ConfigYaml(config_file)

    file_name = config.values['base_folder'] + '/' + get_file_name(config, stage)

    print("Reading the file: " + str(file_name))
    data_sample = pd.read_parquet(file_name)

    data_sample['APtBin'] = pd.cut(data_sample['Pt_a'], config.values['correlation']['bins_assoc'])
    data_sample['TPtBin'] = pd.cut(data_sample['Pt_t'], config.values['correlation']['bins_trig'])

    return data_sample


def save_pairs(data_sample, config_file, stage='raw'):
    config = configyaml.ConfigYaml(config_file)
    file_name = get_file_name(config, stage)
    file_name = config.values['base_folder'] + '/' + file_name

    print("Saving the file to: " + str(file_name))
    data_sample.loc[:, data_sample.columns[data_sample.dtypes != 'category']].to_parquet(file_name)


def get_run_list(configuration_name):
    file_list = glob.glob(storage_location + configuration_name + "/*" + "event.parquet")
    run_list = [get_friendly_parquet_file_name(file, 'event') for file in file_list]
    return run_list


def get_file_list(configuration_name, particle, step='raw'):
    if step == 'raw':
        return glob.glob(storage_location + configuration_name + "/*" + particle + ".parquet")
    elif step == 'filtered':
        return glob.glob(definitions.PROCESSING_FOLDER + configuration_name + '/filtered/' + "/*" + particle +
                         ".parquet")
    elif step == 'filtered':
        pass
    return list()


def load(configuration_name, particle,
         step='raw',
         run_number=None, columns=None,
         index=None, sample_factor=None,
         lazy=False):
    """Loads the dataset from the default storage location. If run_number is a list, all the runs in the list will be
    merged.

    Parameters
    ----------
    configuration_name: str
        The name of the configuration. It can be obtained by checking the name of the folder inside file_name. It should
        start with the value set to base_folder_name. It is a parameter in the AddTask.

    particle: str or list
        The particle name, such as ``electron` or ``dmeson``. The same name that was used to save it.

    step

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

    if run_number is None:
        # Find all the runs if no run is set
        file_list = glob.glob(storage_location + configuration_name + "/*" + particle[0] + ".parquet")
        run_list = [get_friendly_parquet_file_name(file, particle[0]) for file in file_list]
        return load(configuration_name, particle, run_number=run_list, columns=columns, index=index,
                    sample_factor=sample_factor, lazy=lazy)

    file_list = [[storage_location + configuration_name + r"/" + str(run) + '_' + x + '.parquet' for x in particle]
                 for run in run_number]

    if sample_factor is not None:
        if sample_factor > 1.:
            raise ValueError("It is not possible to sample for than 1")
        number_to_sample = int(sample_factor * len(file_list))
        print("Sampling only " + str(number_to_sample) + ' out of ' + str(len(file_list)) + ' files')
        file_list = random.sample(file_list, number_to_sample)

    if lazy:
        return [[LazyFileLoader(x, index=index, columns=col) for x, col in zip(list_run, columns)]
                for list_run in file_list]

    data_sets = list()
    for f in file_list:
        for x in f:
            try:
                df = pd.read_parquet(x, columns=columns)
                data_sets.append(df)
            except OSError:
                warnings.warn('It is not possible to load the files with run number = ' + str(x))
                return None

    if len(data_sets) < 0:
        raise ValueError('No data was loaded.')

    result_dataset = pd.concat(data_sets, sort=False)

    if index is not None:
        result_dataset.set_index(index, inplace=True)

    reduce_dataframe_memory(result_dataset)

    # Temporary solution
    if 'InvMassD0' in result_dataset.columns:
        result_dataset['InvMass'] = result_dataset['InvMassD0']
        result_dataset.drop('InvMassD0', axis='columns', inplace=True)

    return result_dataset


def get_friendly_root_file_name(file):
    return file.split('/')[-1][:-5]


def get_friendly_parquet_file_name(file, particle=''):
    return file.split('/')[-1][:-9 - len(particle)]


def reduce_dataframe_memory(df):
    for col in df.columns[df.dtypes == 'float64']:
        df[col] = df[col].astype(np.float32)

    for col in df.columns[df.dtypes == 'int64']:
        df[col] = df[col].astype(np.int32)

    return df
