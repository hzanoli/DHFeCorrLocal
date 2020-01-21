#!/usr/bin/env python
import warnings

import pandas as pd

import dhfcorr.io.data_reader as reader
import dhfcorr.selection.selection as sl


def expand_array_cols(df):
    for col in df.columns[df.dtypes == object]:
        df_unpack = pd.DataFrame(df[col].values.tolist())
        df_unpack.columns = [col + str(i) for i in range(len(df_unpack.columns))]
        df = pd.concat([df, df_unpack], axis='columns').drop(col, axis='columns')

    return df


def preprocess(file_name, file_identifier, configuration_name, configuration_name_save=None, save=True,
               particle_list=('electron', 'dmeson', 'event')):
    print("Current file: " + file_name)

    try:
        electron, d_meson, events = reader.read_root_file(file_name, configuration_name, particle_list)
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

        reader.save(electron, configuration_name_save, particle_list[0], file_identifier)
        reader.save(d_meson, configuration_name_save, particle_list[1], file_identifier)
        reader.save(events, configuration_name_save, particle_list[2], file_identifier)

    else:
        return electron, d_meson, events


def convert_root_to_parquet(configuration_name_root, local_dataset_name, file_list_root):
    file_list_root = list(pd.read_csv(file_list_root)['file'])
    print("The following files will be processed: " + str(file_list_root))
    print("With configuration name on ROOT: " + configuration_name_root)
    print("With dataset name (local disk): " + local_dataset_name)
    print()
    print("----------- Starting the conversion -----------")
    print()
    for root_file in file_list_root:
        identifier = root_file.split('/')[-1].split('.root')[0]
        preprocess(root_file, identifier, configuration_name_root, local_dataset_name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("configuration_name", help='Name of the configuration of the task. This should be same same '
                                                   'value used in the task on GRID. Check the tree name.')
    parser.add_argument("dataset_name", help="Name of the local dataset.")
    parser.add_argument("file_list", help="csv file with the files that will be processed.")

    args = parser.parse_args()

    convert_root_to_parquet(args.configuration_name, args.dataset_name, args.file_list)
