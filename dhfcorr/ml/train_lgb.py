#!/usr/bin/env python

import argparse
import os
import shutil
import time

import lightgbm as lgb
import pandas as pd

import dhfcorr.config_yaml as configyaml
import dhfcorr.io.data_reader as dr


def train_model(dataset_name, pt_bin, yaml_file, prefix):
    d_cuts = configyaml.ConfigYaml(yaml_file)

    train = dr.get_ml_dataset(dataset_name, d_cuts, pt_bin)

    params = d_cuts.values['model_building']['model_parameters']
    train_parameters = d_cuts.values['model_building']['train_parameters']

    cv_params = d_cuts.values['model_building']['cv_parameters']
    cv_params.update(train_parameters)

    features = d_cuts.values['model_building']['features']
    target = d_cuts.values['model_building']['target']

    lgb_dataset = lgb.Dataset(train[features], label=train[target])

    del train

    start = time.time()
    cv = lgb.cv(params, lgb_dataset, **cv_params)
    print('Total CV time: ' + str(time.time() - start))
    results_cv = pd.DataFrame(cv)

    cv_results_file = dr.get_location_step(dataset_name, 'ml') + 'cv_' + str(pt_bin) + '.pkl'

    try:
        os.remove(cv_results_file)
    except FileNotFoundError:
        pass

    print('Best iteration of the model: ')
    print(results_cv.iloc[-1])
    results_cv.to_pickle(cv_results_file)

    train_parameters['num_boost_round'] = len(results_cv)

    start = time.time()
    gbm = lgb.train(params, lgb_dataset, **train_parameters)
    print('Total training time: ' + str(time.time() - start))

    name_to_save = dr.get_location_step(dataset_name, 'ml') + prefix + 'model_' + str(pt_bin) + '.txt'

    try:
        os.remove(name_to_save)
    except FileNotFoundError:
        pass

    temp_file = dr.definitions.TEMP + 'temp_model.txt'

    gbm.save_model(temp_file)
    shutil.copyfile(temp_file, name_to_save)

    os.remove(temp_file)

    return gbm, name_to_save


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pt_bin", help="Number of the pT bin that will be trained")
    parser.add_argument("config", help="config")
    parser.add_argument("--prefix", default='', help='Prefix when saving the model files')
    parser.add_argument("--yaml_file", default=None, help='YAML file with the configurations of the analysis. If None, '
                                                          'uses the default configuration.')
    args = parser.parse_args()

    main_model, file_list = train_model(args.config, args.pt_bin, args.yaml_file, args.prefix)
