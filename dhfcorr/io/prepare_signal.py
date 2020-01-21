import os
import subprocess

import numpy as np
import pandas as pd

from dhfcorr.io import data_reader as dr
from dhfcorr.selection import selection as sl


def calculate_mean_sigma_mc(df):
    mc_shape = df.groupby(['PtBin'])['InvMass'].agg({'mean', 'std', 'count'}).reset_index()
    mc_shape['PtMin'] = mc_shape['PtBin'].apply(lambda x: x.left).astype(np.float)
    mc_shape['PtMax'] = mc_shape['PtBin'].apply(lambda x: x.right).astype(np.float)
    mc_shape['PtMid'] = mc_shape['PtBin'].apply(lambda x: x.mid).astype(np.float)
    mc_shape['XErr'] = mc_shape['PtBin'].apply(lambda x: x.length / 2.).astype(np.float)
    mc_shape.set_index('PtBin', inplace=True)
    return mc_shape


def prepare_signal(mc_config, pt_bins, particle):
    print("Processing signal")
    signal = sl.get_true_dmesons(dr.load(mc_config, particle))
    signal['PtBin'] = pd.cut(signal['Pt'], pt_bins)

    # Create the columns which will be used for the ML training 'CandidateType'
    # CandidateType = -1 -> Background
    # CandidateType =  0 -> Non-Prompt D mesons
    # CandidateType =  1 -> Prompt D mesons

    signal['CandidateType'] = signal['IsPrompt'].astype(np.int)

    folder_to_save = dr.get_location_step(mc_config, 'ml')
    dr.check_for_folder(folder_to_save)

    delete_previous = subprocess.Popen('rm -f ' + folder_to_save + 'sig_*', shell=True)
    delete_previous.wait()

    for name, group in signal.groupby('PtBin', as_index=False):
        print(name)
        df = group.drop(['PtBin'], axis='columns')
        df.to_parquet(folder_to_save + 'sig_' + str(name) + '.parquet')

    mc_mean_sigma = calculate_mean_sigma_mc(signal)

    os.remove(folder_to_save + 'mc_mean_sigma.pkl')

    mc_mean_sigma.to_pickle(folder_to_save + 'mc_mean_sigma.pkl')
