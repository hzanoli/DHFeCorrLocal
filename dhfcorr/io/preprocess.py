#!/usr/bin/env python3

import dhfcorr.io.data_reader as reader
import dhfcorr.selection.selection as sl
import pandas as pd
import warnings


def expand_array_cols(df):
    for col in df.columns[df.dtypes == object]:
        df_unpack = pd.DataFrame(df[col].values.tolist())
        df_unpack.columns = [col + str(i) for i in range(len(df_unpack.columns))]
        df = pd.concat([df, df_unpack], axis='columns').drop(col, axis='columns')

    return df


def preprocess(file_name, identifier, configuration_name, configuration_name_to_save=None, save=True):
    electron, d_meson = reader.read_root_file(file_name, configuration_name)
    d_meson = expand_array_cols(d_meson)
    sl.build_additional_features_dmeson(d_meson)
    sl.build_additional_features_electron(electron)

    reader.reduce_dataframe_memory(d_meson)
    reader.reduce_dataframe_memory(electron)

    if save:
        if configuration_name_to_save is None:
            configuration_name_to_save = configuration_name
        reader.save(electron, configuration_name_to_save, 'electron', identifier)
        reader.save(d_meson, configuration_name_to_save, 'd_meson', identifier)
    else:
        return electron, d_meson


if __name__ == '__main__':
    """"Save the ROOT files to parquet files. The additional features are also added.
    The first argument should be the name of the configuration.
    The second argument should be the folder that contains the root files to be converted.
    The third argument should be the folder that the files will be saved.
    """

    import sys
    import glob

    config = str(sys.argv[1])
    print('Configuration name = ' + config)

    files = glob.glob(sys.argv[2] + "/*.root")
    periods = [x.split('/')[-1][:-5] for x in files]

    print('Processing the following periods/runs:')
    print(periods)

    try:
        reader.set_storage_location(sys.argv[3])
    except IndexError:
        warnings.warn('No location to save the parquet files provided. The default one will be used.')

    print('Saving them to:' + reader.storage_location)

    for per, f in zip(periods, files):
        print('Current period = ' + str(per))
        preprocess(f, per, config)
