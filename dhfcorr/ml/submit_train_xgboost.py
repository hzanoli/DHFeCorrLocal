#!/usr/bin/env python3

import numpy as np
import pandas as pd
import argparse
import dhfcorr.config_yaml as configyaml
import dhfcorr.definitions as definitions
import subprocess
from dhfcorr.submit_job import get_job_command


if __name__ == '__main__':
    print("Train the model. All are submitted with qsub.")

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

    for i in reversed(range(len(pt_bins))):
        arguments = str(i) + ' ' + str(args.config)
        if args.prefix is not None:
            arguments += ' --prefix ' + args.prefix

        command = get_job_command('pt_' + str(i), base_f + "/ml/train_xgboost.py ", arguments, queue='gpu7')
        # 'qsub -q gpu7 -j oe -V -N pt_' + str(i) + ' ' + '-F ' + arguments + ' ' \
        #      + base_f + "/ml/train_xgboost.py "
        # command = str('qsub -S $(which python3) -pe openmpi 4 -cwd -V -N pt_' + str(
        #    i) + ' ' + base_f + "/ml/train_xgboost.py " + str(i) + ' ' + str(args.config))
        # if args.yaml_file is not None:
        #    command += ' --yaml_file ' + args.yaml_file
        # if args.prefix is not None:
        #    command += ' --prefix ' + args.prefix
        print(command)
        subprocess.run(command, shell=True)
