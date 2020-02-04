#!/usr/bin/env python3

import argparse
import subprocess

import pandas as pd
from tqdm import tqdm

import dhfcorr.definitions as definitions
import dhfcorr.io.data_reader as dr
from dhfcorr.cluster import get_job_command
from dhfcorr.io.data_reader import check_for_folder
from dhfcorr.utils import batch


def split_submit_job(files, script_path, job_name_pattern, n_files, additional_argumets='', test=True):
    print(files)
    if test:
        files = files[:10]
        n_files = 2

    job_id = 0

    for file_list in tqdm(batch(files, n_files), total=int(len(files) / n_files) + 1,
                          desc='Submitting jobs: '):
        job_name = job_name_pattern + str(job_id)
        pd.DataFrame({'file': file_list}).to_csv(job_name + '.csv')
        arguments = ' ' + job_name + '.csv ' + additional_argumets
        command = get_job_command(job_name, script_path, arguments)
        subprocess.run(command, shell=True)

        job_id = job_id + 1

    return len(files)


def submit_predict(dataset_name, particle, n_files, prefix=None, yaml_file=None):
    check_for_folder(dr.get_location_step(dataset_name, 'consolidated'))
    files = dr.find_missing_processed_files(dataset_name, 'raw', 'consolidated', particle, full_file_path=True)
    print(files)

    additional_arguments = ''
    if prefix is not None:
        additional_arguments += ' --prefix ' + str(prefix)
    if yaml_file is not None:
        additional_arguments += ' --yaml_file ' + str(yaml_file)

    n_files_to_process = split_submit_job(files, definitions.ROOT_DIR + '/ml/predict.py', dataset_name + '_p_', n_files,
                                          additional_arguments)

    return n_files_to_process


if __name__ == '__main__':
    print("Predicts the classes in data. All the work is submitted to the cluster.")

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", help="Name of the configuration")
    parser.add_argument("--particle", default='dmeson', help="Name of the particle")
    parser.add_argument("--yaml_file", default=None, help='YAML file with the configurations of the analysis.')
    parser.add_argument("--prefix", default=None, help='Prefix when saving the model files')
    parser.add_argument('-n', "--nfiles", type=int, help='Number of runs per job.', default=20)

    args = parser.parse_args()

    submit_predict(args.dataset_name, args.particle, args.nfiles, args.prefix)
