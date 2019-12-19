#!/usr/bin/env python3

import argparse
import subprocess

import dhfcorr.definitions as definitions
import dhfcorr.io.data_reader as reader
from dhfcorr.io.data_reader import check_for_folder
from dhfcorr.submit_job import get_job_command

if __name__ == '__main__':
    print("Predicts the classes in data. All the work is submitted to the cluster.")

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of the configuration")
    parser.add_argument("--particle", default='dmeson', help="Name of the particle")
    parser.add_argument("--config_to_save", default=None, help="Configuration name that will be used to be saved. If "
                                                               "None is provided, the name will be the same")
    parser.add_argument("--yaml_file", default=None, help='YAML file with the configurations of the analysis.')
    parser.add_argument("--prefix", default=None, help='Prefix when saving the model files')
    parser.add_argument('-n', "--nfiles", type=int, help='Number of runs per job.', default=1)
    parser.add_argument('-c', '--continue_previous', dest='search_for_processed', action='store_true')
    parser.add_argument('-d', '--delete_previous', dest='search_for_processed', action='store_false')
    parser.set_defaults(search_for_processed=True)

    args = parser.parse_args()
    from dhfcorr.utils import batch, format_list_to_bash

    processing_folder = definitions.PROCESSING_FOLDER

    check_for_folder(processing_folder + args.config)
    check_for_folder(processing_folder + args.config + '/filtered/')

    if args.config_to_save is not None:
        check_for_folder(processing_folder + args.config_to_save)
        check_for_folder(processing_folder + args.config_to_save + '/filtered/')

    runs = reader.get_run_numbers(args.config)

    print('Total number of runs: ' + str(len(runs)))
    print('Searching for preprocessed files?: ' + str(args.search_for_processed))

    if args.search_for_processed:
        runs = reader.search_for_processed(args.config, 'raw', 'filtered', 'dmeson')

    print('The following runs will be processed: ')
    print(runs)

    job_id = 0
    for run_list in batch(runs, args.nfiles):
        job_name = 'p_' + args.config + '_' + str(job_id)

        script = definitions.ROOT_DIR + '/ml/predict.py'
        arguments = format_list_to_bash(run_list) + ' ' + args.config

        if args.yaml_file is not None:
            arguments += ' --yaml_file ' + args.yaml_file
        if args.prefix is not None:
            arguments += ' --prefix ' + args.prefix
        if args.config_to_save is not None:
            arguments += ' --config_to_save ' + args.config_to_save

        command = get_job_command(job_name, script, arguments)
        print()
        print("Submitting job " + str(job_id))
        subprocess.run(command, shell=True)
        job_id = job_id + 1
