#!/usr/bin/env python

import argparse
import glob

from tqdm import tqdm

import dhfcorr.definitions as definitions
import dhfcorr.io.data_reader as reader
from dhfcorr.io.data_reader import get_run_number
from dhfcorr.submit_job import get_job_command

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
    parser.add_argument('-n', "--n_runs", type=int, help='Number of runs per job.', default=20)
    parser.add_argument('-c', '--continue_previous', dest='search_for_processed', action='store_true')
    parser.add_argument('-d', '--delete_previous', dest='search_for_processed', action='store_false')
    parser.set_defaults(search_for_processed=True)

    args = parser.parse_args()

    dataset_name = args.dataset_name
    name_root = args.configuration_name

    if name_root is None:
        name_root = dataset_name

    print('Configuration (root file) = ' + name_root)
    print('Dataset name (in this system) = ' + dataset_name)

    folder_root_files = definitions.DATA_FOLDER + '/root_merged/' + dataset_name
    print('Folder with root files: ' + folder_root_files)
    files = glob.glob(folder_root_files + "/*.root")
    runs = list({get_run_number(x) for x in files})

    print('Total number of runs of this dataset = ' + str(len(runs)))

    if args.search_for_processed:
        runs = reader.search_for_processed(runs, dataset_name, 'raw')
        print('Total number of runs excluding the ones already processed = ' + str(len(runs)))

    print('Processing the following periods (or runs):')
    print(runs)

    import subprocess
    from dhfcorr.utils import batch

    job_id = 0
    print()
    print("Submitting jobs:")
    for period in tqdm(batch(runs, args.n_runs), total=int(len(runs) / args.n_runs) + 1):
        job_name = 'conv_' + dataset_name + '_' + str(job_id)

        script_path = definitions.ROOT_DIR + '/io/convert_to_parquet.py'
        arguments = name_root + ' ' + dataset_name + ' -r '
        for p in period:
            arguments += str(p) + ' '
        command = get_job_command(job_name, script_path, arguments)
        subprocess.run(command, shell=True, stdout=subprocess.PIPE)
        job_id = job_id + 1
