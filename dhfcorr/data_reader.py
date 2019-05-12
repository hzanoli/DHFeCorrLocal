from __future__ import print_function
import warnings
try: 
    import ROOT
    import root_numpy
except ModuleNotFoundError:
    warnings.warn("ROOT is not installed. Only pandas interface will work.", RuntimeWarning)


import numpy as np
import pandas as pd

path_macros = "/mnt/home_folder/cernbox/postdoc/task-D0-HFe-correlation/"

types_in_tree = dict( {'electron': 'AliDHFeCorr::AliElectronTree', 'dmeson': 'AliDHFeCorr::AliDMesonTree'})
tree_name = dict( {'electron': 'electron', 'dmeson': 'dmeson'})

min_pid_number = 1459163000

default_saving_location = "data/"
base_folder_name = "DHFeCorrelation_"

def convert_to_pandas(file_name,folder_name,tree_name,branch):
    file_root = ROOT.TFile(file_name)
    ROOT.gDirectory.cd(folder_name)
    tree = ROOT.gDirectory.Get(tree_name)

    df = pd.DataFrame(root_numpy.tree2array(tree,branches=branch))
    
    # remove f from variables
    df.columns = [x[1:] if x.startswith("f") else x for x in df.columns]
    # temporary solution to rename the InvMass
    df.columns = ["InvMassD0" if x== "InvMass" else x for x in df.columns]
    df.columns = ["InvMassD0bar" if x== "InvMassAnti" else x for x in df.columns]

    df['GridPID'] = df['GridPID'] - min_pid_number
    df['GridPID'] = pd.to_numeric(df['GridPID'], downcast='integer')
    
    file_root.Close()
    return df

def get_features(particle_of_interesst, task_name = "AliAnalysisTaskDHFeCorr"):
    if (particle_of_interesst not in types_in_tree.keys()):
        raise ValueError('The particle of interested is not found. Given values: '+str(particle_of_interesst))
    try:
        task = getattr(ROOT,task_name)
    except:
        try:
            ROOT.gInterpreter.ProcessLine(".include $ROOTSYS/include")
            ROOT.gInterpreter.ProcessLine(".include $ALICE_ROOT/include")
            ROOT.gInterpreter.LoadMacro(path_macros+task_name+".cxx++g")
            task = getattr(ROOT,task_name)
        except:
            raise ValueError("The task with task_name="+task_name+" is not found and it was not possible to compile it")

    #task = getattr(ROOT,task_name)
    variables = getattr(ROOT,types_in_tree[particle_of_interesst]).__dict__.keys()

    #remove the ones comming from python
    variables_purge = [x for x in variables if not x.startswith('__')]

    return variables_purge

def default_read(file_name, configuration_name):
    br_e = ['fPt', 'fGridPID','fEventNumber','fEta', 'fPhi', 'fNClsTPC', 'fNClsTPCDeDx', 'fNITSCls', 'fDCAxy', 'fDCAz', 'fTPCNSigma', 'fTOFNSigma', 'fNULS', 'fNLS']
    br_d = ['fPt', 'fGridPID','fEventNumber','fEta', 'fPhi', 'fY', 'fNormDecayLengthXY', 'fCosP', 'fCosPXY', 'fImpParXY', 'fDCA', 'fNormd0MeasMinusExp', 'fInvMass', 'fInvMassAnti', 'fCosTsD0', 'fCosTsD0bar', 'fPtDaughter0', 'fPtDaughter1', 'fD0Daughter0', 'fD0Daughter1', 'fReducedChi2', 'fSelectionStatus']
    folder_name = base_folder_name + configuration_name

    electrons = convert_to_pandas(file_name, folder_name, tree_name["electron"], br_e)
    dmesons = convert_to_pandas(file_name, folder_name, tree_name["dmeson"], br_d)
    return electrons,dmesons

def save(df,configuration_name,particle,run_number):
    df.to_hdf(default_saving_location+configuration_name+r"/"+str(run_number)+'.h5',particle)
    
def load(configuration_name,particle,run_number):
    return pd.read_hdf(default_saving_location+configuration_name+r"/"+str(run_number)+'.h5',particle)

