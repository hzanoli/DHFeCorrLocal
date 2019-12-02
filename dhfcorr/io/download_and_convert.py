import argparse
import subprocess
import io
import pandas as pd
import os
import sys
import time
import dhfcorr.definitions as definitions
from dhfcorr.io.download_file import get_token


def job_status(job_name_pattern):
    user = os.environ['USER']
    cols = ['Job ID', 'Username', 'Queue', 'Jobname', 'SessID', 'NDS', 'TSK', 'ReqMemory', 'ReqTime', 'Status', 'Time']
    command = 'qstat -u ' + user + ' | grep ' + job_name_pattern
    try:
        result = pd.read_csv(
            io.StringIO(subprocess.run(command, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')),
            delim_whitespace=True, header=None)
    except pd.errors.EmptyDataError:
        return None
    if len(result.columns) != 0:
        result.columns = cols
        return result
    return None


def is_job_complete(job_name_pattern):
    job_status_df = job_status(job_name_pattern)
    if job_status_df is None:
        return True
    uncompleted_jobs = job_status_df[job_status_df['Status'] != 'C']
    if len(uncompleted_jobs) > 0:
        return False
    return True


def get_n_jobs(job_name_pattern):
    job_status_df = job_status(job_name_pattern)
    if job_status_df is None:
        return 0
    uncompleted_jobs = job_status_df[job_status_df['Status'] != 'C']
    return len(uncompleted_jobs)


def wait_jobs_to_finish(step_name, job_name_pattern):
    n_jobs_running = get_n_jobs(job_name_pattern)
    len_message = len('\rWaiting for ' + step_name + ' to finish [running ' + str(n_jobs_running) + ' jobs]')

    while n_jobs_running > 0:
        sys.stdout.write('\rWaiting for ' + step_name + ' to finish [running ' + str(n_jobs_running) + ' jobs]')
        time.sleep(1)

    message_over = '\r' + step_name + ' done!'
    blank_space = str(' ' * (len_message - len(message_over)))[:len_message]

    sys.stdout.write(message_over + blank_space)


if __name__ == '__main__':
    print("Downloading and converting the files.")
    print("This script will call multiple other scripts to get the job done.")
    print()

    parser = argparse.ArgumentParser()
    parser.add_argument("user", help='User on grid')
    parser.add_argument("code", help='Code to unlock the certificate')
    parser.add_argument("train_name", help='Name of the train (eg. HFCJ_pp')
    parser.add_argument("destination", help="Destination of the file that will be downloaded. It is always added to "
                                            "the basic definitions from the definitions.py file")
    parser.add_argument("-r", "--train_runs", help='Number of the run in the Lego train system', nargs='+',
                        required=True)
    args = parser.parse_args()

    print('First, lets download the files.')
    print()

    get_token(args.code)
    files_download = subprocess.Popen(definitions)
    from dhfcorr.io.submit_download_grid import submit_download_grid

    submit_download_grid(args.user, args.code, args.train_name, args.destination, args.train_runs, args.n_files)
