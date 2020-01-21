#!/usr/bin/env python

import argparse

import numpy as np
import pandas as pd

import dhfcorr.config_yaml as configyaml
import dhfcorr.io.data_reader as reader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("run_list", type=str, help='name of the runs that will be read.')
    parser.add_argument("config_name", type=str, help='Configuration name (used to save the temporary files)')
    parser.add_argument("--yaml_config", default=None, help='Configuration file)')
    parser.add_argument("--id", default=0, help='id to save the file')
    parser.add_argument("--particle_name", default='dmeson', help='particle name')

    args = parser.parse_args()
    run_list = args.run_list
    run_list = run_list.split(',')

    yaml_config = args.yaml_config
    d_cuts = configyaml.ConfigYaml(yaml_config)

    folder_to_save = reader.get_location_step('ml')
    mc_mean = pd.read_pickle(folder_to_save + '/mc_mean_sigma.pkl')

    def filter_bkg(df, mc_shape, n_sigma=4.0):
        pt_bin = df.name
        mean = mc_shape.loc[pt_bin]['mean']
        std = mc_shape.loc[pt_bin]['std']
        bkg_sidebands = df[np.abs(df['InvMass'] - mean) > n_sigma * std]
        return bkg_sidebands


    candidates_df = list()
    for run in run_list:
        bkg = reader.load(args.config_name, args.particle_name, run_number=[run])
        bkg['PtBin'] = pd.cut(bkg['Pt'], bins=d_cuts.values['model_building']['bins_pt'])
        candidates = bkg.groupby('PtBin', as_index=False).apply(filter_bkg, mc_shape=mc_mean)
        candidates_df.append(candidates)

    df_merged = pd.concat(candidates_df)

    for name, group in df_merged.groupby('PtBin', as_index=False):
        df_this_bin = group.drop(['PtBin'], axis='columns').reset_index(drop=True)
        if len(df_this_bin) > 0:
            df_this_bin.to_parquet(folder_to_save + 'bkg_' + str(args.id) + '_' + str(name) + '.parquet')
