#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
import glob
import os
import dhfcorr.definitions as definitions


def merge_files(file_list, file_destination):
    total_df = pd.concat([pd.read_parquet(f) for f in file_list])
    total_df.to_parquet(file_destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", help="Name of the local dataset.")
    parser.add_argument("-r", "--runs", help='Number of the runs (or periods) that will be merged ', nargs='+',
                        required=True)

    parser.add_argument("-p", "--particle", help='Particles (and event) that will be merged', nargs='+',
                        required=True)

    parser.add_argument('-s', "--step", default='filtered', help='Step of the analysis that will be merged.')

    args = parser.parse_args()

    dataset = args.dataset_name
    particles = args.particle
    folder_input = definitions.PROCESSING_FOLDER + dataset + '/' + args.step
    folder_output = definitions.PROCESSING_FOLDER + dataset + '/consolidated/'

    if not os.path.isdir(folder_input):
        raise FileNotFoundError('The folder with input files to do exist. It was given: \n' + str(folder_input))

    if not os.path.isdir(folder_output):
        os.mkdir(folder_output)

    print('Dataset name = ' + dataset)

    for particle in particles:
        print('Processing particle: ' + particle)
        for run_number in args.runs:
            run_number = str(run_number)
            print("Merging Run(period): " + run_number)
            files_this_run = glob.glob(folder_input + "/*" + run_number + '*' + particle + "*.parquet")
            print('Total number of files in this run = ' + str(len(files_this_run)))
            size_files = [os.path.getsize(x) for x in files_this_run]
            print("Total Size: {0: .3f} GB".format(np.array(size_files).sum() / (1024. ** 3)))
            destination = folder_output + str(run_number) + '_' + particle + '.parquet'
            merge_files(files_this_run, destination)

    print("Processing done.")
