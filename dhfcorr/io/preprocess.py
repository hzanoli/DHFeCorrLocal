#!/usr/bin/env python3

import dhfcorr.io.data_reader as reader
import numpy as np

if __name__ == '__main__':
    """"Save the ROOT files to parquet files. The additional features are also added.
    The first argument should be the name of the configuration.
    The second argument should be the folder that contains the root files to be converted.
    The third argument should be the folder that the files will be saved.
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

    import dhfcorr.definitions as definitions
    import glob
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("configuration_name", help='Name of the configuration of the task. This should be same same '
                                                   'value used in the task on GRID. Check the tree name .')
    parser.add_argument("--name_root", help='Set in case the name of the configuration of the root file is different'
                                            'from the one in the local file system', default=None)
    parser.add_argument("--nfiles", help='Number of files per job.', default=70)
    args = parser.parse_args()

    config = args.configuration_name
    name_root = args.name_root
    if name_root is None:
        name_root = config
    print('Configuration name (root file) = ' + name_root)
    print('Configuration name (in the file system) = ' + config)

    folder_root_files = definitions.DATA_FOLDER + '/root/' + config
    print('Folder with root files: ' + folder_root_files)
    files = glob.glob(folder_root_files + "/*.root")
    periods = np.unique(np.array([x.split('/')[-1][:3] for x in files]))

    print('The total number of files found is : ' + str(len(files)))
    print()

    print('Processing the following periods (or runs):')
    print(periods)

    print('Saving them to:' + reader.storage_location)
    print('I will submit approx. ' + str(len(files) / args.nfiles) + ' jobs')

    import subprocess
    from dhfcorr.io.utils import batch, format_list_to_bash

    job_id = 0
    for file_list in batch(files, args.nfiles):
        file_names = format_list_to_bash(file_list)
        job_name = 'root_2_pqt_' + str(job_id)

        script_path = definitions.ROOT_DIR + '/io/convert_to_parquet.py'
        submit_part = r"qsub -V -cwd -N " + job_name + " -S $(which python3) " + script_path
        command = submit_part + ' ' + config + ' --name_root ' + name_root + ' ' + file_names
        print("Submitting job " + str(job_id))
        subprocess.run(command, shell=True)
        job_id = job_id + 1

    # for per, f in zip(periods, files):
    #    print('Current period = ' + str(per))
    #    preprocess(f, per, config)
