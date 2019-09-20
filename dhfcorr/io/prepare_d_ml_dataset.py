import dhfcorr.io.data_reader as reader
import dhfcorr.selection.selection as sl
import dhfcorr.config_yaml as configyaml
import numpy as np
import pandas as pd
import dhfcorr.definitions as definitions


def process_mc(configuration_mc, columns):
    d_mesons_mc = reader.load(configuration_mc, 'dmeson', columns=columns)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("mc_config", help="Name of the configuration used in MC (used for signal).")
parser.add_argument("data_config", help="Name of the configuration used in data (used for background).")
parser.add_argument("--meson", choices=['D0', 'D+', 'Dstar'], default='D0', help='D meson that will be used.')
parser.add_argument("--yaml_file", default=None, help='YAML file with the configurations of the analysis.')

args = parser.parse_args()

d_cuts = configyaml.ConfigYaml(args.yaml_file)

cols_to_load = d_cuts.values['model_building']['features'] + d_cuts.values['model_building']['additional_features']
cols_to_load_mc = cols_to_load + d_cuts.values['model_building']['additional_features_mc']
signal = sl.get_true_dmesons(reader.load(args.mc_config, 'dmeson'))[cols_to_load_mc]

signal['PtBin'] = pd.cut(signal['Pt'], d_cuts.values['model_building']['bins_pt'])

# Create the columns which will be used for the ML training 'CandidateType'
# CandidateType = -1 -> Background
# CandidateType =  0 -> Non-Prompt D mesons
# CandidateType =  1 -> Prompt D mesons
signal['CandidateType'] = signal['IsPrompt'].astype(np.int)


def calculate_mean_sigma_mc(df):
    mc_shape = df.groupby(['PtBin'])['InvMass'].agg({'mean', 'std', 'count'}).reset_index()
    mc_shape['PtMin'] = mc_shape['PtBin'].apply(lambda x: x.left).astype(np.float)
    mc_shape['PtMax'] = mc_shape['PtBin'].apply(lambda x: x.right).astype(np.float)
    mc_shape['PtMid'] = mc_shape['PtBin'].apply(lambda x: x.mid).astype(np.float)
    mc_shape['XErr'] = mc_shape['PtBin'].apply(lambda x: x.length / 2.).astype(np.float)
    mc_shape.set_index('PtBin', inplace=True)
    return mc_shape


mc_mean_sigma = calculate_mean_sigma_mc(signal)
print(mc_mean_sigma)

import os

processing_folder = definitions.PROCESSING_FOLDER + args.data_config
if not os.path.isdir(processing_folder):
    os.mkdir(processing_folder)

mc_mean_sigma.to_pickle(processing_folder + '/mc_mean_sigma.pkl')

# Load data for Background
# bkg = reader.load(args.data_config, 'dmeson')[cols_to_load]
# bkg['PtBin'] = pd.cut(bkg['Pt'], bins=d_cuts.values['model_building']['bins_pt'])
# bkg.dropna('index', inplace=True)
