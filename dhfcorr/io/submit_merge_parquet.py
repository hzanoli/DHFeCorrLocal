import argparse

from tqdm import tqdm

import dhfcorr.definitions as definitions
import dhfcorr.io.data_reader as reader
from dhfcorr.cluster import get_job_command

if __name__ == '__main__':
    print('Merging parquet files for a single run/period')

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", help='Name of the local dataset')

    parser.add_argument('-r', '--by-run', dest='merge_period', action='store_false', help='Perform merging by run')
    parser.add_argument('-p', '--by-period', dest='merge_period', action='store_true', help='Perform merging by period')
    parser.set_defaults(merge_period=True)

    args = parser.parse_args()

    dataset_name = args.dataset_name

    print('Dataset name = ' + dataset_name)

    if args.merge_period:
        runs = reader.get_periods(dataset_name, step='filtered')
    else:
        runs = reader.get_run_numbers(dataset_name, step='filtered')

    print("The following periods/runs will be processed: ")
    print(runs)

    import subprocess

    job_id = 0
    print()
    print("Submitting jobs:")

    for run in tqdm(runs):
        job_name = 'm_pt_' + dataset_name + '_' + str(run)
        script_path = definitions.ROOT_DIR + '/io/merge_parquet_files.py'

        arguments_d = dataset_name + ' -s filtered -p dmeson -r ' + str(run)
        command_d = get_job_command(job_name + '_d', script_path, arguments_d)
        subprocess.run(command_d, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        args_ele_ev = dataset_name + ' -s raw -p electron event -r ' + str(run)
        command_ele_ev = get_job_command(job_name + '_e', script_path, args_ele_ev)
        subprocess.run(command_ele_ev, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        job_id = job_id + 1
