#!/usr/bin/env python3

import argparse
import subprocess

import numpy as np
import pandas as pd

import dhfcorr.config_yaml as configyaml
import dhfcorr.definitions as definitions
from dhfcorr.cluster import get_job_command


def submit_train(dataset_name, yaml_config, prefix=None):
    d_cuts = configyaml.ConfigYaml(yaml_config)

    pt_bins = np.array(d_cuts.values['model_building']['bins_pt'])
    pt_bins = pd.cut(0.5 * (pt_bins[:-1] + pt_bins[1:]), bins=pt_bins)
    base_f = definitions.ROOT_DIR

    queue = d_cuts.values['model_building']['queue']

    for i in reversed(range(len(pt_bins))):
        arguments = str(i) + ' ' + str(dataset_name)
        if prefix is not None:
            arguments += ' --prefix ' + prefix

        command = get_job_command(dataset_name + '_t_pt_' + str(i), base_f + "/ml/train_lgb.py ", arguments,
                                  queue=queue)
        subprocess.run(command, shell=True)


if __name__ == '__main__':
    print("Submitting the jobs to train the model")

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", help="Name of the dataset")
    parser.add_argument("--yaml_file", default=None, help='YAML file with the configurations of the analysis.')
    parser.add_argument("--prefix", default=None, help='Prefix when saving the model files')

    args = parser.parse_args()

    submit_train(args.dataset_name, args.yaml_file, args.prefix)
