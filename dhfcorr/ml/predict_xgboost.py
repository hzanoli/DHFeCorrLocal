#!/usr/bin/env python

import argparse
import itertools

import h2o
import numpy as np
import pandas as pd

import dhfcorr.config_yaml as configyaml
import dhfcorr.definitions as definitions
import dhfcorr.io.data_reader as reader


def add_prediction(df, models):
    df = pd.DataFrame({'Prediction': h2o.mojo_predict_pandas(df, models[int(df.name)])['1'].values}, index=df.index)
    return df


def predict_class(files, config, yaml_file, prefix, config_save, test_mode):
    if config_save is None:
        config_save = config

    d_cuts = configyaml.ConfigYaml(yaml_file)
    pt_bins = np.array(d_cuts.values['model_building']['bins_pt'])

    base_name = definitions.PROCESSING_FOLDER + config + '/ml-dataset/' + prefix + 'model_pt'

    models = [base_name + str(pt_bin) + '_main_mojo.zip' for pt_bin in range(len(pt_bins) - 1)]

    for file in files:
        print('Processing file: ')
        print(file)
        dataset = pd.read_parquet(file)

        if test_mode:
            dataset = dataset.iloc[:1000]

        dataset['Probability'] = -999.
        pt_bins_df = pd.cut(dataset['Pt'], list(pt_bins), labels=False)

        predictions = dataset.groupby(pt_bins_df, as_index=False, group_keys=False).apply(add_prediction, models)
        dataset['Probability'] = predictions.astype('float32')

        file_name = file.split('/')[-1]
        dataset.to_parquet(definitions.PROCESSING_FOLDER + config_save + '/filtered/' + file_name)
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", help="run list")
    parser.add_argument("config", help="Configuration name")
    parser.add_argument("--config_to_save", default=None, help="Configuration name")
    parser.add_argument("--particle", default='dmeson', help="Particle name that will be reduced")
    parser.add_argument("--prefix", default='', help='Prefix when saving the model files')
    parser.add_argument("--yaml_file", default=None, help='YAML file with the configurations of the analysis. If None, '
                                                          'uses the default configuration.')
    parser.add_argument('-t', "--test_mode", action='store_true', dest='test',
                        help='Test the script. Process only one file')
    parser.set_defaults(test=False)

    args = parser.parse_args()
    print("Predicting for runs: ")
    run_list = args.runs.split(',')
    print(run_list)

    file_list = list(itertools.chain.from_iterable(
        [reader.find_missing_processed_files(args.config, 'raw', 'filtered', args.particle, run, full_file_path=True)
         for run in run_list]))

    file_list.sort()
    if args.test:
        file_list = file_list[:1]
    print('Processing the files: ')
    for f in file_list:
        print(f)

    predict_class(file_list, args.config, args.yaml_file, args.prefix, args.config_to_save, args.test)
    print('Processing done.')

    # h2o.cluster().shutdown()
