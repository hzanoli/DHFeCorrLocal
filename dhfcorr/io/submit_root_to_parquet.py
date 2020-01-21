#!/usr/bin/env python

import argparse
import subprocess

import pandas as pd
from tqdm import tqdm

import dhfcorr.definitions as definitions
import dhfcorr.io.data_reader as reader
from dhfcorr.cluster import get_job_command


def submit_root_to_parquet(dataset_name, name_root, n_files):
    if name_root is None:
        name_root = dataset_name

    print('Configuration (root file) = ' + name_root)
    print('Dataset name (in this system) = ' + dataset_name)

    folder_root_files = definitions.DATA_FOLDER + '/root_merged/' + dataset_name
    print('Folder with root files: ' + folder_root_files)
    files = reader.find_missing_processed_files(dataset_name, 'root_merged', 'raw', None, full_file_path=True)

    from dhfcorr.utils import batch

    job_id = 0

    print()
    print("")

    for file_list in tqdm(batch(files, n_files), total=int(len(files) / n_files) + 1, desc='Submitting jobs: '):
        job_name = dataset_name + '_conv_' + str(job_id)

        script_path = definitions.ROOT_DIR + '/io/convert_to_parquet.py'
        arguments = name_root + ' ' + dataset_name + ' ' + job_name + '.csv'

        pd.DataFrame({'file': file_list}).to_csv(job_name + '.csv')

        command = get_job_command(job_name, script_path, arguments)
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        job_id = job_id + 1

    return len(files)


if __name__ == '__main__':
    """"Save the ROOT files to parquet files. The additional features are also added.
    """
    # image from http://patorjk.com/software/taag/#p=display&f=Slant&t=ROOT%20-%3E%20Parquet%0A
    print("\n"
          r"    ____  ____  ____  ______     __       ____                              __ " "\n"
          r"   / __ \/ __ \/ __ \/_  __/     \ \     / __ \____ __________ ___  _____  / /_" "\n"
          r"  / /_/ / / / / / / / / /    _____\ \   / /_/ / __ `/ ___/ __ `/ / / / _ \/ __/" "\n"
          r" / _, _/ /_/ / /_/ / / /    /_____/ /  / ____/ /_/ / /  / /_/ / /_/ /  __/ /_  " "\n"
          r"/_/ |_|\____/\____/ /_/          / /  /_/    \__,_/_/   \__, /\__,_/\___/\__/  " "\n"
          r"                                /_/                        /_/                  " "\n"
          )

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", help='Name of the local dataset')
    parser.add_argument("configuration_name", help='Name of the configuration in the ROOT file.')
    parser.add_argument('-n', "--n_files", type=int, help='Number of files per job.', default=20)
    parser.add_argument('-c', '--continue_previous', dest='search_for_processed', action='store_true')
    parser.add_argument('-d', '--delete_previous', dest='search_for_processed', action='store_false')
    parser.set_defaults(search_for_processed=True)

    args = parser.parse_args()

    submit_root_to_parquet(args.dataset_name, args.configuration_name, args.n_files)
