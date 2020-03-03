#!/usr/bin/env python
import warnings

import pandas as pd

import dhfcorr.io.data_reader as dr
import dhfcorr.selection.selection as sl


def expand_array_cols(df):
    for col in df.columns[df.dtypes == object]:
        df_unpack = pd.DataFrame(df[col].values.tolist())
        df_unpack.columns = [col + str(i) for i in range(len(df_unpack.columns))]
        df = pd.concat([df, df_unpack], axis='columns').drop(col, axis='columns')

    return df


def preprocess(file_name, file_identifier, configuration_name, configuration_name_save=None, save=True,
               particle_list=('electron', 'dmeson', 'event'), n_threads=None):
    print("Current file: " + file_name)

    try:
        dfs = dr.read_root_file(file_name, configuration_name, particle_list, return_dict=True, n_threads=n_threads)
    except OSError:
        warnings.warn('File ' + file_name + ' has one or more invalid trees (likely empty')
        return None

    if 'dmeson' in particle_list:
        dfs['dmeson'] = expand_array_cols(dfs['dmeson'])

    if 'electron' in particle_list:
        sl.build_additional_features_electron(dfs['electron'])

    for key, value in dfs.items():
        dr.reduce_dataframe_memory(value)

    if save:
        if configuration_name_save is None:
            configuration_name_save = configuration_name

        for key, value in dfs.items():
            dr.save(value, configuration_name_save, key, file_identifier)

    else:
        return [value for key, value in dfs.items()]


def convert_root_to_parquet(configuration_name_root, local_dataset_name, file_list_root,
                            particle_list=('electron', 'dmeson', 'event')):
    file_list_root = list(pd.read_csv(file_list_root)['file'])
    print("The following files will be processed: " + str(file_list_root))
    print("With configuration name on ROOT: " + configuration_name_root)
    print("With dataset name (local disk): " + local_dataset_name)
    print()
    print("----------- Starting the conversion -----------")
    print()
    for root_file in file_list_root:
        identifier = root_file.split('/')[-1].split('.root')[0]
        preprocess(root_file, identifier, configuration_name_root, local_dataset_name, particle_list=particle_list)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("configuration_name", help='Name of the configuration of the task. This should be same same '
                                                   'value used in the task on GRID. Check the tree name.')
    parser.add_argument("dataset_name", help="Name of the local dataset.")
    parser.add_argument("file_list", help="csv file with the files that will be processed.")
    parser.add_argument("-p", "--particle_list", nargs='*', required=False, default=[('electron', 'dmeson', 'event')],
                        help="csv file with the files that will be processed.")

    args = parser.parse_args()

    convert_root_to_parquet(args.configuration_name, args.dataset_name, args.file_list)
