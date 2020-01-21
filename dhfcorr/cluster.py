import io
import os
import subprocess
import sys
import time

import pandas as pd

from dhfcorr import definitions as definitions


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


def is_job_completed(job_name_pattern):
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
    len_message = len('\rWaiting for: ' + step_name + ' [running ' + str(n_jobs_running) + ' jobs]')

    while n_jobs_running > 0:
        n_jobs_running = get_n_jobs(job_name_pattern)
        sys.stdout.write('\rWaiting for: ' + step_name + ' [running ' + str(n_jobs_running) + ' jobs]')
        time.sleep(1)

    message_over = '\r' + step_name + ' done!'
    blank_space = str(' ' * (len_message - len(message_over)))[:len_message]

    sys.stdout.write(message_over + blank_space)


def get_job_command(name, script, arguments=None, cluster='stbc', queue=definitions.JOB_QUEUE):
    if cluster == 'stbc':
        qsub_cmd = 'qsub '
        if arguments is not None:
            qsub_cmd += '-F \"' + arguments + '\"'
        qsub_cmd += ' -q ' + queue + ' -j oe -V -N ' + name
        qsub_cmd += ' ' + script
        return qsub_cmd

    elif cluster == 'quark':
        qsub_args = '-V -cwd'

    elif cluster == 'local':
        cmd = 'python ' + script
        if arguments is not None:
            cmd = ' ' + str(arguments)
        return cmd
    else:
        raise ValueError("The selected cluster is not defined.")


def get_token(code_token, user_name=''):
    token = subprocess.Popen('echo ' + code_token + ' | alien-token-init ' + user_name, shell=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    token.wait()
    time.sleep(1)
