import argparse
import dhfcorr.io.data_reader as reader
import dhfcorr.config_yaml as configyaml
import dhfcorr.definitions as definitions
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm


def get_valid_events(df, identifier=('GridPID', 'EventNumber')):
    return df.set_index(identifier).index.unique()


def match_assoc_trig_events(file, unique_ids, identifier=('GridPID', 'EventNumber')):
    return pd.read_parquet(file).set_index(identifier).loc(unique_ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Configuration name")
    parser.add_argument("--particle", default='electron', help="Name of the particle")
    parser.add_argument("--yaml_file", default=None, help='YAML file with the configurations of the analysis. If None, '
                                                          'uses the default configuration.')
    args = parser.parse_args()

    file_list = reader.get_file_list(args.config, args.particle, step='raw')

    base_name = definitions.PROCESSING_FOLDER + args.config + '/skimmed/'
