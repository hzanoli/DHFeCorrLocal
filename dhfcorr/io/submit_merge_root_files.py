#!/usr/bin/env python

import argparse
import glob
import os
import subprocess

from tqdm import tqdm

import dhfcorr.definitions as definitions
from dhfcorr.cluster import get_job_command
from dhfcorr.io.data_reader import get_run_number
from dhfcorr.utils import batch


def submit_merge_root_files(dataset_name, maximum_size, continue_pre, number_of_runs):
    print('Dataset name = ' + dataset_name)

    folder_root_files = definitions.DATA_FOLDER + '/root/' + dataset_name
    folder_to_save = definitions.DATA_FOLDER + '/root_merged/' + dataset_name

    if not os.path.isdir(definitions.DATA_FOLDER + '/root_merged/'):
        os.mkdir(definitions.DATA_FOLDER + '/root_merged/')

    if not os.path.isdir(folder_to_save):
        os.mkdir(folder_to_save)
    elif not continue_pre:  # Delete previous merge
        delete_files = subprocess.Popen('rm -f ' + folder_to_save + '/*', shell=True)
        delete_files.wait()

    print('Folder with root files: ' + folder_root_files)
    print('Target file size(GB): ' + str(maximum_size))

    files = glob.glob(folder_root_files + "/*.root")
    periods = {get_run_number(x) for x in files}

    print('The total number of files found is : ' + str(len(files)))
    print()
    print('Saving them to:' + folder_to_save)

    if continue_pre:
        print("The merge will not delete files from previous iterations. Only new periods will be reprocessed.")
        files_already_merged = glob.glob(folder_to_save + "/*.root")

        period_already_merged = {get_run_number(x) for x in files_already_merged}
        if len(period_already_merged) == 0:
            print("No previous merged files.")
        else:
            print("Found " + str(len(period_already_merged)) + "periods already merged")
            print(period_already_merged)
        periods = periods - period_already_merged

    periods = list(periods)
    if len(periods) == 0:
        print("No periods to merge")

    job_id = 0

    for period in tqdm(list(batch(periods, number_of_runs)), desc='Submitting jobs'):
        job_name = dataset_name + '_merge_' + str(job_id)
        arguments = str(dataset_name) + ' --target_sizeGB ' + str(maximum_size) + ' -r '
        for p in period:
            arguments = arguments + str(p) + ' '
        command = get_job_command(job_name, definitions.ROOT_DIR + '/io/merge_root_files.py', arguments)
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        job_id += 1

    return len(periods)


if __name__ == '__main__':
    print("Merging root files")

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", help="Name of the local dataset.")
    parser.add_argument('-t', "--target_sizeGB", default=1., help='Maximum file size for merged files.')
    parser.add_argument('-c', '--continue_previous_merge', dest='continue_merge', action='store_true')
    parser.add_argument('-d', '--delete_previous_merge', dest='continue_merge', action='store_false')
    parser.add_argument('-n', '--number_of_runs', type=int, default=100, help='Number of runs per job')
    parser.set_defaults(continue_merge=True)

    args = parser.parse_args()

    dataset = args.dataset_name
    max_size = args.target_sizeGB
    continue_previous = args.continue_merge

    submit_merge_root_files(args.dataset_name, args.target_sizeGB, args.continue_merge)
