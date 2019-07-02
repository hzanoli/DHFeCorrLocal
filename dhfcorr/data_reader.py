from __future__ import print_function
import warnings

try:
    import ROOT
    import root_numpy
except ModuleNotFoundError:
    warnings.warn("ROOT is not installed. Only pandas interface will work.", RuntimeWarning)

import pandas as pd

path_macros = "/mnt/home_folder/cernbox/postdoc/task-D0-HFe-correlation/"

types_in_tree = dict({'electron': 'AliDHFeCorr::AliElectronTree', 'dmeson': 'AliDHFeCorr::AliDMesonTree'})
tree_name = dict({'electron': 'electron', 'dmeson': 'dmeson'})

min_pid_number = 1459163000

default_saving_location = "data/"
base_folder_name = "DHFeCorrelation_"


def convert_to_pandas(file_name, folder_name, tree_name, branch=None):
    file_root = ROOT.TFile(file_name)
    ROOT.gDirectory.cd(folder_name)
    tree = ROOT.gDirectory.Get(tree_name)
    if branch is not None:
        df = pd.DataFrame(root_numpy.tree2array(tree, branches=branch))
    else:
        df = pd.DataFrame(root_numpy.tree2array(tree))

    # remove f from variables
    df.columns = [x[1:] if x.startswith("f") else x for x in df.columns]
    # temporary solution to rename the InvMass
    df.columns = ["InvMassD0" if x == "InvMass" else x for x in df.columns]
    df.columns = ["InvMassD0bar" if x == "InvMassAnti" else x for x in df.columns]

    df['GridPID'] = df['GridPID'] - min_pid_number
    df['GridPID'] = pd.to_numeric(df['GridPID'], downcast='integer')

    file_root.Close()
    return df


def get_features(particle_of_interest, task_name="AliAnalysisTaskDHFeCorr"):
    if particle_of_interest not in types_in_tree.keys():
        raise ValueError('The particle of interested is not found. Given values: ' + str(particle_of_interest))
    try:
        getattr(ROOT, task_name)
    except AttributeError:
        try:
            ROOT.gInterpreter.ProcessLine(".include $ROOTSYS/include")
            ROOT.gInterpreter.ProcessLine(".include $ALICE_ROOT/include")
            ROOT.gInterpreter.LoadMacro(path_macros + task_name + ".cxx++g")
            getattr(ROOT, task_name)
        except AttributeError:
            raise ValueError(
                "The task with task_name=" + task_name + " is not found and it was not possible to compile it")

    variables = getattr(ROOT, types_in_tree[particle_of_interest]).__dict__.keys()

    # Remove the ones coming from python
    variables_purge = [x for x in variables if not x.startswith('__')]

    return variables_purge


def read_root_file(file_name, configuration_name):
    """Read the root file with name file_name (should include the path to the file) and configuration named
    configuration_name. Returns DataFrame with the contents of the associated ROOT.TTree.

    Parameters
    ----------
    file_name : str
        Name of the file that contains the ROOT TTree. Should contain the path to the file.

    configuration_name: str
        The name of the configuration. It can be obtained by checking the name of the folder inside file_name. It should
        start with the value set to base_folder_name. It is the only parameter in the AddTask.

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


def default_read(file_name, configuration_name):
    br_e = ['fPt', 'fGridPID', 'fEventNumber', 'fEta', 'fPhi', 'fNClsTPC', 'fNClsTPCDeDx', 'fNITSCls', 'fDCAxy',
            'fDCAz', 'fTPCNSigma', 'fTOFNSigma', 'fNULS', 'fNLS']
    br_d = ['fPt', 'fGridPID', 'fEventNumber', 'fEta', 'fPhi', 'fY', 'fNormDecayLengthXY', 'fCosP', 'fCosPXY',
            'fImpParXY', 'fDCA', 'fNormd0MeasMinusExp', 'fInvMass', 'fInvMassAnti', 'fCosTsD0', 'fCosTsD0bar',
            'fPtDaughter0', 'fPtDaughter1', 'fD0Daughter0', 'fD0Daughter1', 'fReducedChi2', 'fSelectionStatus']

    folder_name = base_folder_name + configuration_name

    electrons = convert_to_pandas(file_name, folder_name, tree_name["electron"], br_e)
    dmesons = convert_to_pandas(file_name, folder_name, tree_name["dmeson"], br_d)
    return electrons, dmesons


def save(df, configuration_name, particle, run_number):
    import os
    if not os.path.isdir(default_saving_location + configuration_name):
        os.mkdir(default_saving_location + configuration_name)

    # df.to_hdf(default_saving_location + configuration_name + r"/" + str(run_number) + '.h5', particle)

    df.to_parquet(default_saving_location + configuration_name + r"/" + str(run_number) + '_' + particle + '.parquet',
                  index=True)


def load_hdf(configuration_name, particle, run_number):
    if isinstance(run_number, (str, int, float)):
        run_number = [run_number]

    data_sets = list()

    for run in run_number:
        try:
            df = pd.read_hdf(default_saving_location + configuration_name + r"/" + str(run) + '.h5', particle)
            data_sets.append(df)
        except FileNotFoundError:
            warnings.warn('It is not possible to load the files with run number = ' + str(run))
            return None

    if len(data_sets) < 0:
        raise ValueError('No data was loaded.')

    result_dataset = pd.concat(data_sets)

    return result_dataset


def load(configuration_name, particle, run_number):
    if isinstance(run_number, (str, int, float)):
        run_number = [run_number]

    data_sets = list()

    for run in run_number:
        try:
            # df = pd.read_hdf(default_saving_location + configuration_name + r"/" + str(run) + '.h5', particle)
            df = pd.read_parquet(
                default_saving_location + configuration_name + r"/" + str(run) + '_' + particle + '.parquet')
            data_sets.append(df)
        except OSError:
            warnings.warn('It is not possible to load the files with run number = ' + str(run))
            return None

    if len(data_sets) < 0:
        raise ValueError('No data was loaded.')

    result_dataset = pd.concat(data_sets)

    return result_dataset
