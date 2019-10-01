import argparse
import dhfcorr.config_yaml as configyaml
import dhfcorr.definitions as definitions
import numpy as np
import pandas as pd
import glob
import os
import random
from tqdm import tqdm
import subprocess


def get_signal_df(folder, pt_bin_name, cols):
    file = folder + 'sig_' + pt_bin_name + '.parquet'
    df = pd.read_parquet(file)[cols].copy()
    return df.reset_index(drop=True)


def get_background_df(folder, pt_bin_name, cols):
    files = glob.glob(folder + 'bkg_*' + pt_bin_name + '*.parquet')
    total_file_size = (np.array([os.path.getsize(file) for file in files])).sum() / (1024 ** 3)
    print('Found a total of: {} files with approx. {:0.1f} GB'.format(len(files), total_file_size))

    if total_file_size > 25.:
        sample_rate = 25. / total_file_size
        files = random.sample(files, int(sample_rate * len(files)))
        total_file_size = (np.array([os.path.getsize(file) for file in files])).sum() / (1024 ** 3)
        print('Too large files, so I will sample only GB'.format(total_file_size))

    print("Reading background files: ")
    bkg_dfs = [pd.read_parquet(file, use_threads=True)[cols] for file in tqdm(files)]
    merged_background = pd.concat(bkg_dfs, ignore_index=True, sort=False).reset_index(drop=True)

    return merged_background.reset_index(drop=True)


def build_dataframe_ml(folder, pt_bin_name, cols, cols_mc):
    signal = get_signal_df(folder, pt_bin_name, cols_mc)

    background = get_background_df(folder, pt_bin_name, cols)
    if len(background) > 2 * len(signal):
        background = background.sample(2 * len(signal))
    background['CandidateType'] = -1

    total_df = pd.concat([signal, background], ignore_index=True).reset_index(drop=True)
    return total_df


if __name__ == '__main__':
    print("Merging the data samples")

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of the configuration")
    parser.add_argument("--yaml_file", default=None, help='YAML file with the configurations of the analysis.')
    parser.add_argument("--remove_intermediate", type=bool, default=False, help='Remove the intermediate stages')

    args = parser.parse_args()
    d_cuts = configyaml.ConfigYaml(args.yaml_file)

    pt_bins = np.array(d_cuts.values['model_building']['bins_pt'])
    pt_bin_names = [str(x) for x in pd.cut(0.5 * (pt_bins[:-1] + pt_bins[1:]), bins=pt_bins)]

    folder_saved = definitions.PROCESSING_FOLDER + args.config + '/ml-dataset/'

    cols_to_load = d_cuts.values['model_building']['features'] + d_cuts.values['model_building']['additional_features']
    cols_to_load_mc = cols_to_load + ['CandidateType']

    print()
    for pt_bin, i in zip(pt_bin_names, range(len(pt_bin_names))):
        print('Processing bin: ' + pt_bin)
        merged = build_dataframe_ml(folder_saved, pt_bin, cols_to_load, cols_to_load_mc)
        merged.to_parquet(folder_saved + 'ml_sample_' + i + '.parquet')
        print()

    if args.remove_intermediate:
        subprocess.run('rm ' + folder_saved + 'bkg*')
        subprocess.run('rm ' + folder_saved + 'sig*')
