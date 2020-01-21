import argparse
import glob
import os

from dhfcorr.cluster import wait_jobs_to_finish, get_token


def clean_job_submitted(pattern):
    files = glob.glob(pattern)
    for f in files:
        os.remove(f)


if __name__ == '__main__':
    print("Downloading files from grid and converting to parquet.")

    parser = argparse.ArgumentParser()

    parser.add_argument("user", help='User on grid')
    parser.add_argument("code", help='Code to unlock the certificate')
    parser.add_argument("train_name", help='Name of the train (eg. HFCJ_pp')
    parser.add_argument("destination", help="Destination of the file that will be downloaded. It is always added to "
                                            "the basic definitions from the definitions.py file")
    parser.add_argument("configuration_root", help="Name of the configuration in the ROOT files.")
    parser.add_argument("-r", "--train_runs", help='Number of the run in the Lego train system', nargs='+',
                        required=True)

    parser.add_argument('-t', "--target_sizeGB", default=1., type=float, help='Maximum file size for merged files.')

    parser.add_argument("-n", "--n_files", help='Number of the files to be processed at the same job', default=50)
    parser.add_argument('-nr', "--n_runs", type=int, help='Number of runs per job for parquet conversion.',
                        default=10)

    args = parser.parse_args()
    print()
    print('Downloading files')

    get_token(args.code, args.user)

    from dhfcorr.io.submit_download_grid import submit_download_grid

    n_files_to_download = 1
    while n_files_to_download > 0:
        n_files_to_download = submit_download_grid(args.user, args.code, args.train_name, args.destination,
                                                   args.train_runs, args.n_files)
        wait_jobs_to_finish(' download files', args.destination + '_d_')

    print('Finished downloading the files!')
    print()

    # Cleaning the files used to submit
    clean_job_submitted(args.destination + '_d_*')
    os.remove('login.csv')

    print('Merging the ROOT files')

    from dhfcorr.io.submit_merge_root_files import submit_merge_root_files

    n_runs_to_convert = 1
    while n_runs_to_convert > 0:
        n_runs_to_convert = submit_merge_root_files(args.destination, args.target_sizeGB, True, args.n_runs)
        wait_jobs_to_finish('merge files', args.destination + '_merge_')

    print('Finished merging root files!')
    print()

    # Cleaning the files used to submit
    clean_job_submitted(args.destination + '_merge_*')

    print('Converting to parquet')

    from dhfcorr.io.submit_root_to_parquet import submit_root_to_parquet

    max_trials = 3
    current_trial = 0
    n_files_to_convert = 1
    while n_files_to_convert > 0 and current_trial < max_trials:
        n_files_to_convert = submit_root_to_parquet(args.destination, args.configuration_root, args.n_runs)
        wait_jobs_to_finish('convert files', args.destination + '_conv_')
        current_trial += 1

    if current_trial == max_trials:
        print("It was not possible to convert all the ROOT files to parquet. This can happen sometimes there are no "
              "candidates in the file. Double check it to be sure.")

    print('Converted all the files!')
    print()

    clean_job_submitted(args.destination + '_conv_*')

    print('All the steps are finished.')
