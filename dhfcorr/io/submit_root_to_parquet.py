#!/usr/bin/env python

import dhfcorr.io.data_reader as reader
import dhfcorr.definitions as definitions
import glob
import argparse
from dhfcorr.io.data_reader import get_run_number
from dhfcorr.submit_job import get_job_command
from tqdm import tqdm


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
    parser.add_argument('-n', "--n_runs", help='Number of runs per job.', default=20)

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
    periods = list({get_run_number(x) for x in files})

    print('The total number of files found is : ' + str(len(files)))
    print()

    print('Processing the following periods (or runs):')
    print(periods)

    print('Saving them to:' + reader.storage_location)

    import subprocess
    from dhfcorr.io.utils import batch

    job_id = 0
    print()
    print("Submitting jobs:")
    for period in tqdm(batch(periods, args.n_runs), total=int(len(periods) / args.n_runs) + 1):
        job_name = 'conv_' + dataset_name + '_' + str(job_id)

        script_path = definitions.ROOT_DIR + '/io/convert_to_parquet.py'
        arguments = name_root + ' ' + dataset_name + ' -r '
        for p in period:
            arguments += str(p) + ' '
        command = get_job_command(job_name, script_path, arguments)
        subprocess.run(command, shell=True, stdout=subprocess.PIPE)
        job_id = job_id + 1
