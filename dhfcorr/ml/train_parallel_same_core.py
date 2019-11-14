#!/usr/bin/env python3

import numpy as np
import pandas as pd
import argparse
import dhfcorr.config_yaml as configyaml
import dhfcorr.definitions as definitions
import subprocess
import h2o

if __name__ == '__main__':
    print("Train the model. All are submitted with qsub.")

    h2o.init(max_mem_size_GB=int(definitions.CLUSTER_MEMORY))

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of the configuration")
    parser.add_argument("--yaml_file", default=None, help='YAML file with the configurations of the analysis.')
    parser.add_argument("--prefix", default=None, help='Prefix when saving the model files')

    args = parser.parse_args()

    # Global model configuration
    d_cuts = configyaml.ConfigYaml(args.yaml_file)

    pt_bins = np.array(d_cuts.values['model_building']['bins_pt'])
    pt_bins = pd.cut(0.5 * (pt_bins[:-1] + pt_bins[1:]), bins=pt_bins)
    base_f = definitions.ROOT_DIR

    max_time = d_cuts.values['model_building']['max_time_cluster']
    queue = d_cuts.values['model_building']['queue']

    processes = list()
    for i in [9, 10, 11, 12]:  # reversed(range(len(pt_bins))):
        arguments = str(i) + ' ' + str(args.config)
        if args.prefix is not None:
            arguments += '--prefix ' + args.prefix
        command = './' + base_f + "/ml/train_xgboost.py " + arguments

        print(command)
        subprocess.Popen(command, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
