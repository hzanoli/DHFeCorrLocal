#!/usr/bin/env python

import dhfcorr.io.data_reader as reader
import os
import glob
import numpy as np
import argparse
import pandas as pd
import dhfcorr.config_yaml as configyaml
import dhfcorr.definitions as definitions
from tqdm import tqdm
from dhfcorr.io.utils import batch
from dhfcorr.io.data_reader import reduce_dataframe_memory


def reduce_opt(files_to_reduce, config, yaml_file, id_job, particle, pre_filter_bkg, maximum_pt_filter):
    d_cuts = configyaml.ConfigYaml(yaml_file)
    pt_bins = np.array(d_cuts.values['reduce_data']['bins_pt'])
    cols_keep = d_cuts.values['reduce_data']['features']

    base_name = definitions.PROCESSING_FOLDER + config + '/skimmed/'

    dataset = pd.concat([pd.read_parquet(file, columns=cols_keep) for file in files_to_reduce])

    dataset = dataset.loc[((dataset['bkg'] < pre_filter_bkg) & (dataset['Pt'] < maximum_pt_filter)) | (
            dataset['Pt'] >= maximum_pt_filter)]

    reduce_dataframe_memory(dataset)
    df_pt_bins = pd.cut(dataset['Pt'], list(pt_bins), labels=False)
    dataset.groupby(df_pt_bins).apply(
        lambda x: x.to_parquet(base_name + 'id' + str(id_job) + '_pt' + str(x.name) + '_' + particle + '.parquet'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Configuration name")
    parser.add_argument("--particle", default='dmeson', help="Name of the particle")
    parser.add_argument("--pre_filter_bkg", default=0.5, help='Maximum probability of the background that will be '
                                                              'still accepted')
    parser.add_argument("--maximum_pt_filter", default=3., help='Maximum pT that pre_filter_bkg will  be used')
    parser.add_argument("--yaml_file", default=None, help='YAML file with the configurations of the analysis. If None, '
                                                          'uses the default configuration.')
    parser.add_argument("--nfiles", help='Number of files per job.', default=30)

    args = parser.parse_args()

    file_list = reader.get_file_list(args.config, args.particle, step='filtered')
    print('Skimming the files')

    total_file_size = (np.array([os.path.getsize(file) for file in file_list])).sum() / (1024 ** 3)
    print('Size before skimming: {:0.2f} GB'.format(total_file_size))

    processing_folder = definitions.PROCESSING_FOLDER + args.config
    if not os.path.isdir(processing_folder):
        os.mkdir(processing_folder)
    if not os.path.isdir(processing_folder + '/skimmed'):
        os.mkdir(processing_folder + '/skimmed/')

    file_batches = list(batch(file_list, args.nfiles))
    for files, job_id in tqdm(zip(file_batches, range(len(file_batches))), total=len(file_batches)):
        reduce_opt(files, args.config, args.yaml_file, job_id, args.particle, args.pre_filter_bkg,
                   args.maximum_pt_filter)

    files_produced = glob.glob(definitions.PROCESSING_FOLDER + args.config + '/skimmed/*' + args.particle + '.parquet')
    size_after = (np.array([os.path.getsize(file) for file in files_produced])).sum() / (1024 ** 3)

    print('Size after skimming: {:0.2f} GB (reduction {:0.2f} times)'.format(size_after, total_file_size / size_after))

    print('Processing done.')
