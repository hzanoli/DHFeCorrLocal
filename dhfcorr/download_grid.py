#!/usr/bin/env python

import subprocess


def run_command(command):
    subprocess.run(command.split())


def init_token(username):
    run_command('alien-token-init ' + username)


def download_file(grid_path, local_path):
    run_command('alien_cp  alien:' + str(grid_path) + ' file:' + local_path)


def download_opt(username, runs, grid_folder, local_folder, is_data=True, file_name='AnalysisResults.root'):
    if isinstance(runs, (int, float)):
        runs = [runs]
    else:
        try:
            iter(runs)
        except TypeError as tp:
            raise TypeError('Run list is not iterable')

    init_token(username)
    base_folder_name = '/alice/cern.ch/user/' + username[0] + '/' + username + '/'
    base_folder_name += grid_folder + '/'

    for run in runs:
        path = base_folder_name + str(run)

        if is_data:
            path = base_folder_name + '000' + str(run) + '/'

        download_file(path + file_name, local_folder + '/' + str(run) + '.root')
