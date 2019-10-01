import numpy as np
import pandas as pd
import argparse
import dhfcorr.config_yaml as configyaml
import dhfcorr.definitions as definitions
from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.grid.grid_search import H2OGridSearch

import h2o

if __name__ == '__main__':
    print("Train the model. The first pt bins are trained in the main nodes and the others are submitted with SGE.")

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of the configuration")
    parser.add_argument("--yaml_file", default=None, help='YAML file with the configurations of the analysis.')
    parser.add_argument("--n_cores", default=None, help='Number of cores used to submit the jobs')
    args = parser.parse_args()

    # Global model configuration
    d_cuts = configyaml.ConfigYaml(args.yaml_file)

    pt_bins = np.array(d_cuts.values['model_building']['bins_pt'])
    pt_bins = pd.cut(0.5 * (pt_bins[:-1] + pt_bins[1:]), bins=pt_bins)

    folder_saved = definitions.PROCESSING_FOLDER + args.config + '/ml-dataset/'
    features = d_cuts.values['model_building']['features']
    target = 'CandidateType'

    parameter_grid = d_cuts.values['model_building']['grid_search']
    n_folds = d_cuts.values['model_building']['nfolds']

    h2o.init(max_mem_size='16G')
    pt_bin = 0
    config = 'D0_HMV0'

    dataset = h2o.import_file(
        definitions.PROCESSING_FOLDER + config + '/ml-dataset/ml_sample_' + str(pt_bin) + '.parquet')

    dataset[target] = dataset[target].asfactor()

    train, test = dataset.split_frame([0.5], seed=1234)

    grid = H2OGridSearch(model=H2OXGBoostEstimator,
                         grid_id='grid_' + str(pt_bin),
                         hyper_params=parameter_grid)

    model = H2OXGBoostEstimator()
    # grid.train(features, target,
    #           training_frame=train,
    #           nfolds=n_folds,
    #           seed=1234)

    model.train(features, target, training_frame=train)

    # h2o.cluster().shutdown()
