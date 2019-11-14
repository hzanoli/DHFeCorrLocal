#!/usr/bin/env python

import h2o
import argparse
import dhfcorr.config_yaml as configyaml
import os
import dhfcorr.definitions as definitions
from h2o.estimators.xgboost import H2OXGBoostEstimator
import sys


def train_model(config, pt_bin, yaml_file, prefix):
    train_f = definitions.PROCESSING_FOLDER + config + '/ml-dataset/ml_sample_train_' + str(pt_bin) + '.parquet'
    train = h2o.import_file(train_f)

    d_cuts = configyaml.ConfigYaml(yaml_file)

    # Configuration of the GRID Search
    features = d_cuts.values['model_building']['features']
    target = d_cuts.values['model_building']['target']
    parameters = d_cuts.values['model_building']['model_parameters']

    train[target] = train[target].asfactor()

    model = H2OXGBoostEstimator(**parameters)

    model.train(features, target, training_frame=train)

    place_to_save = definitions.PROCESSING_FOLDER + config + '/ml-dataset/'
    file_list_saved = list()

    # Save Main model
    path_main = h2o.save_model(model, place_to_save, force=True)
    path_main_rename = ''.join([x + '/' for x in path_main.split('/')[:-1]]) + prefix + 'model_pt' + str(
        pt_bin) + '_main'
    os.rename(path_main, path_main_rename)
    file_list_saved.append(path_main_rename)

    model_list = model.cross_validation_models()
    for model_cv, i in zip(model_list, range(len(model_list))):
        path = h2o.save_model(model_cv, place_to_save, force=True)
        path_new = ''.join([x + '/' for x in path.split('/')[:-1]]) + prefix + 'model_pt' + str(pt_bin) + '_cv' + str(i)
        os.rename(path, path_new)
        file_list_saved.append(path_new)

    return model, model_list, file_list_saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pt_bin", help="Number of the pT bin that will be trained")
    parser.add_argument("config", help="config")
    parser.add_argument("--prefix", default='', help='Prefix when saving the model files')
    parser.add_argument("--yaml_file", default=None, help='YAML file with the configurations of the analysis. If None, '
                                                          'uses the default configuration.')
    args = parser.parse_args()
    # setup_h2o_cluster_sge(str(args.pt_bin) + args.prefix)
    # h2o.connect()
    h2o.init(max_mem_size_GB=int(definitions.CLUSTER_MEMORY), nthreads=6)
    main_model, cv_models, file_list = train_model(args.config, args.pt_bin, args.yaml_file, args.prefix)
    h2o.cluster().shutdown()

    if not all([os.path.isfile(x) for x in file_list]):
        sys.exit(1)
