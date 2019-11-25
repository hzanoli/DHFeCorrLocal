#!/usr/bin/env python

import h2o
import numpy as np
import tqdm
import argparse
import pandas as pd
import dhfcorr.config_yaml as configyaml
import dhfcorr.definitions as definitions
import dhfcorr.io.data_reader as reader


def predict_class(files, config, yaml_file, prefix, config_save):
    if config_save is None:
        config_save = config

    d_cuts = configyaml.ConfigYaml(yaml_file)
    pt_bins = np.array(d_cuts.values['model_building']['bins_pt'])

    base_name = definitions.PROCESSING_FOLDER + config + '/ml-dataset/' + prefix + 'model_pt'

    models = list()
    for pt_bin in range(len(pt_bins) - 1):
        models.append(h2o.load_model(base_name + str(pt_bin) + '_main'))

    for file in files:
        print('Processing file: ')
        print(file)
        dataset = h2o.import_file(file)
        dataset['PtBin'] = dataset['Pt'].cut(list(pt_bins))
        bins_names = dataset['PtBin'].categories()
        map_bins = dict(zip(bins_names, models))

        filtered = list()
        for pt_bin in bins_names:
            model = map_bins[pt_bin]
            this_bin_data = dataset[dataset['PtBin'] == pt_bin]
            predictions = model.predict(this_bin_data)
            this_bin_data['bkg'] = predictions['p0']
            # this_bin_data['nonprompt'] = predictions['p0']
            # this_bin_data['prompt'] = predictions['p1']
            filtered.append(this_bin_data.drop('PtBin').as_data_frame())

        combined = pd.concat(filtered, ignore_index=True)
        file_name = file.split('/')[-1]

        combined.to_parquet(definitions.PROCESSING_FOLDER + config_save + '/filtered/' + file_name)

        h2o.remove(dataset)
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
    args = parser.parse_args()

    h2o.init(max_mem_size_GB=definitions.CLUSTER_MEMORY)
    print("Predicting for runs: ")
    run_list = args.runs.split(',')
    print(run_list)
    file_list = reader.get_files_from_runlist(args.config, run_list, args.particle)

    print('Processing the files: ')
    for f in file_list:
        print(f)

    predict_class(file_list, args.config, args.yaml_file, args.prefix, args.config_to_save)
    print('Processing done.')
    h2o.cluster().shutdown()
