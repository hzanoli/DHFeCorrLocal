#!/usr/bin/env python

import argparse
import numpy as np
import glob
import os
import dhfcorr.definitions as definitions
import subprocess
from tqdm import tqdm

from dhfcorr.io.data_reader import split_files, get_period


def merge_root_files(file_list, target_name):
    argument = ''
    for file in file_list:
        argument += ' ' + str(file)

    merger = subprocess.Popen('hadd {0} {1}'.format(str(target_name), argument), shell=True, stdout=subprocess.PIPE)
    merger.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", help="Name of the local dataset.")
    parser.add_argument("--target_sizeGB", default=1., help='Maximum total size of the merged files.')
    parser.add_argument("-r", "--runs", help='Number of the runs that will be merged ', nargs='+', required=True)

    args = parser.parse_args()

    dataset = args.dataset_name

    if not os.path.isdir(definitions.DATA_FOLDER + '/root_merged/'):
        os.mkdir(definitions.DATA_FOLDER + '/root_merged/')

    print('Dataset name = ' + dataset)
    folder_root_files = definitions.DATA_FOLDER + '/root/' + dataset
    folder_to_save = definitions.DATA_FOLDER + '/root_merged/' + dataset + '/'

    if not os.path.isdir(folder_to_save):
        os.mkdir(folder_to_save)
    for run_number in args.runs:
        run_number = str(run_number)
        print("Merging Run: " + run_number)
        files_this_run = glob.glob(folder_root_files + "/*" + run_number + "*.root")
        print('Total number of files in this run = ' + str(len(files_this_run)))
        size_files = [os.path.getsize(x) for x in files_this_run]
        print("Total Size: {0: .3f} GB".format(np.array(size_files).sum() / (1024. ** 3)))

        period = str(get_period(files_this_run[0]))
        print("Period: " + period)

        file_groups = split_files(files_this_run, size_files, float(args.target_sizeGB) * (1024 ** 3))

        print("Merging files:")
        for group, i in tqdm(zip(file_groups, range(len(file_groups))), total=len(file_groups)):
            destination = folder_to_save + str(period) + '_' + str(run_number) + '_' + str(i) + '.root'
            merge_root_files(group, destination)

    print("Processing done.")
