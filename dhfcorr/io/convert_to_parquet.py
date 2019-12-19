#!/usr/bin/env python
import glob
import itertools
import warnings

import pandas as pd

import dhfcorr.definitions as definitions
import dhfcorr.io.data_reader as reader
import dhfcorr.selection.selection as sl


def expand_array_cols(df):
    for col in df.columns[df.dtypes == object]:
        df_unpack = pd.DataFrame(df[col].values.tolist())
        df_unpack.columns = [col + str(i) for i in range(len(df_unpack.columns))]
        df = pd.concat([df, df_unpack], axis='columns').drop(col, axis='columns')

    return df


def preprocess(file_name, identifier, configuration_name, configuration_name_save=None, save=True):
    print("Current file: " + file_name)
    try:
        electron, d_meson, events = reader.read_root_file(file_name, configuration_name,
                                                          ['electron', 'dmeson', 'event'])
    except OSError:
        warnings.warn('File ' + file_name + ' has one or more invalid trees (likely empty')
        return None
    d_meson = expand_array_cols(d_meson)
    sl.build_additional_features_dmeson(d_meson)
    sl.build_additional_features_electron(electron)

    reader.reduce_dataframe_memory(d_meson)
    reader.reduce_dataframe_memory(electron)
    reader.reduce_dataframe_memory(events)

    if save:
        if configuration_name_save is None:
            configuration_name_save = configuration_name
        reader.save(electron, configuration_name_save, 'electron', identifier)
        reader.save(d_meson, configuration_name_save, 'dmeson', identifier)
        reader.save(events, configuration_name_save, 'event', identifier)

    else:
        return electron, d_meson, events


def convert_root_to_parquet(configuration_name, configuration_name_save, root_files):
    print("The following files will be processed: " + str(root_files))
    print("With configuration name on ROOT: " + configuration_name)
    print("With dataset name (local disk): " + configuration_name_save)
    print()
    print("----------- Starting the conversion -----------")
    print()
    for root_file in root_files:
        identifier = root_file.split('/')[-1].split('.root')[0]
        preprocess(root_file, identifier, configuration_name, configuration_name_save)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("configuration_name", help='Name of the configuration of the task. This should be same same '
                                                   'value used in the task on GRID. Check the tree name.')
    parser.add_argument("dataset_name", help="Name of the local dataset.")
    parser.add_argument("-r", "--run_number", help='Number of the runs with will be converted', nargs='+',
                        required=True)

    args = parser.parse_args()
    dataset = args.dataset_name

    folder_merged_root_files = definitions.DATA_FOLDER + '/root_merged/' + dataset

    files = [glob.glob(folder_merged_root_files + "/*" + run + "*.root") for run in args.run_number]
    files = list(itertools.chain.from_iterable(files))

    convert_root_to_parquet(args.configuration_name, args.dataset_name, files)
