#!/usr/bin/env python3

import argparse
import os
import dhfcorr.definitions as definitions
import subprocess
import dhfcorr.io.data_reader as reader


def check_for_folder(folder):
    if folder is None:
        return
    if not os.path.isdir(folder):
        os.mkdir(folder)


if __name__ == '__main__':
    print("Predicts the classes in data. All the work is submitted with SGE.")

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of the configuration")
    parser.add_argument("--particle", default='dmeson', help="Name of the particle")
    parser.add_argument("--config_to_save", default=None, help="Configuration name that will be used to be saved. If "
                                                               "None is provided, the name will be the same")
    parser.add_argument("--yaml_file", default=None, help='YAML file with the configurations of the analysis.')
    parser.add_argument("--prefix", default=None, help='Prefix when saving the model files')
    parser.add_argument("--nfiles", help='Number of files per job.', default=70)
    parser.add_argument("-s", "--search_for_processed", help='Search for processed files before submitting jobs.'
                                                             'Processed files that are found will not be predicted '
                                                             'Useful if some jobs failed.', default=False)

    args = parser.parse_args()
    from dhfcorr.io.utils import batch, format_list_to_bash

    processing_folder = definitions.PROCESSING_FOLDER

    check_for_folder(processing_folder + args.config)
    check_for_folder(processing_folder + args.config + '/filtered/')
    check_for_folder(processing_folder + args.config_to_save)
    check_for_folder(processing_folder + args.config_to_save + '/filtered/')

    files = reader.get_file_list(args.config, args.particle)
    print(len(files))
    print(args.search_for_processed)
    if args.search_for_processed:
        config = args.config
        if args.config_to_save is not None:
            config = args.config_to_save
        files_processed = reader.get_file_list(config, args.particle, step='filtered')
        files = [f for f in files if f not in files_processed]

    job_id = 0
    for file_list in batch(files, args.nfiles):
        job_name = 'ml_pred_' + str(job_id)

        submit_part = "qsub -V -cwd -N " + job_name + " -S $(which python3) " + definitions.ROOT_DIR + \
                      '/ml/predict.py'
        command = submit_part + ' ' + format_list_to_bash(file_list) + ' ' + args.config

        if args.yaml_file is not None:
            command += ' --yaml_file ' + args.yaml_file
        if args.prefix is not None:
            command += ' --prefix ' + args.prefix
        if args.config_to_save is not None:
            command += ' --config_to_save ' + args.config_to_save

        print()
        print("Submitting job " + str(job_id))
        print(command)
        subprocess.run(command, shell=True)
        job_id = job_id + 1
