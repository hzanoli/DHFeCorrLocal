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


from __future__ import print_function
import warnings

try:
    import ROOT
    import root_numpy
except ModuleNotFoundError:
    warnings.warn("ROOT is not installed. Only pandas interface will work.", RuntimeWarning)

import pandas as pd

tree_name = dict({'electron': 'electron', 'dmeson': 'dmeson'})

storage_location = "data/"
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


def convert_to_pandas(file_name, folder_name, tree_name_local, branch=None):
    file_root = ROOT.TFile(file_name)
    ROOT.gDirectory.cd(folder_name)
    tree = ROOT.gDirectory.Get(tree_name_local)

    if branch is not None:
        df = pd.DataFrame(root_numpy.tree2array(tree, branches=branch))
    else:
        df = pd.DataFrame(root_numpy.tree2array(tree))

    # temporary solution to rename the InvMass
    df.columns = ["InvMassD0" if x == "InvMass" else x for x in df.columns]
    df.columns = ["InvMassD0bar" if x == "InvMassAnti" else x for x in df.columns]

    df['GridPID'] = df['GridPID']
    df['GridPID'] = pd.to_numeric(df['GridPID'], downcast='integer')

    file_root.Close()
    return df


def read_root_file(file_name, configuration_name):
    """Read the root file with name file_name (should include the path to the file) and configuration named
    configuration_name. Returns DataFrame with the contents of the associated ROOT.TTree.

    Parameters
    ----------
    file_name : str
        Name of the file that contains the ROOT TTree. Should contain the path to the file.

    configuration_name: str
        The name of the configuration. It can be obtained by checking the name of the folder inside file_name. It should
        start with the value set to base_folder_name. It is a parameter in the AddTask.

    Returns
    -------
    electrons : pd.DataFrame
        DataFrame with the electron data
    dmesons : pd.DataFrame
        DataFrame with the D meson data

    """

    folder_name = base_folder_name + configuration_name

    electrons = convert_to_pandas(file_name, folder_name, tree_name["electron"])
    dmesons = convert_to_pandas(file_name, folder_name, tree_name["dmeson"])

    return electrons, dmesons


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

    df.to_parquet(storage_location + configuration_name + r"/" + str(run_number) + '_' + particle + '.parquet',
                  index=False)


def load(configuration_name, particle, run_number):
    """Loads the dataset from the default storage location. If run_number is a list, all the runs in the list will be
    merged.

    Parameters
    ----------
    configuration_name: str
        The name of the configuration. It can be obtained by checking the name of the folder inside file_name. It should
        start with the value set to base_folder_name. It is a parameter in the AddTask.

    particle: str
        The particle name, such as ``electron` or ``dmeson``. The same name that was used to save it.

    run_number: str
        This is a unique identifier for each file. Usually the run number is used.

    Warnings
    ----------
    UserWarning
        In case one of the run data is not loaded.

    Raises
    ----------
    ValueError
        If no data is loaded.

    """
    if isinstance(run_number, (str, int, float)):
        run_number = [run_number]

    data_sets = list()

    for run in run_number:
        try:
            df = pd.read_parquet(
                storage_location + configuration_name + r"/" + str(run) + '_' + particle + '.parquet')
            data_sets.append(df)
        except OSError:
            warnings.warn('It is not possible to load the files with run number = ' + str(run))
            return None

    if len(data_sets) < 0:
        raise ValueError('No data was loaded.')

    result_dataset = pd.concat(data_sets)

    return result_dataset
