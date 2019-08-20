#!/usr/bin/env python
import subprocess


def run_command(command):
    return subprocess.run(command.split())


def init_token(username):
    return run_command('alien-token-init ' + username)


def download_file(grid_path, local_path):
    return run_command('alien_cp  alien:' + str(grid_path) + ' file:' + local_path)


def get_train_output_file_name(folder_path, file_name):
    command = 'alien_find ' + folder_path + ' ' + file_name
    files = subprocess.run(command.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
    files = files.split('\n')[:-3]
    files = [x.strip() for x in files]
    return files


def get_friendly_file_name(file_name, train_name):
    list_path = file_name.split('/')
    part_with_child_name = list_path[list_path.index(train_name) + 1]

    stage_folder = [x for x in list_path if x.startswith('Stage')]
    value_to_append = ''
    if len(stage_folder) > 0:
        value_to_append = list_path[list_path.index(stage_folder[0]) + 1]

    return 'child' + part_with_child_name.split('child')[1] + '_' + value_to_append


def download_train_opt(folder_path_alien, local_folder, file_name, train_name, period_base):
    files = get_train_output_file_name(folder_path_alien, file_name)
    print('Files found:')

    for x in files:
        print(x)

    friendly_names = [get_friendly_file_name(x, train_name) for x in files]

    print()
    print('Saving files to local directory: ' + local_folder)
    for grid_file, local_file in zip(files, friendly_names):
        local_file = period_base + '_' + local_file + '.root'
        print(grid_file + ' -> ' + local_file)
        download_file(grid_file, local_folder + '/' + local_file)


def download_opt(username, runs, grid_folder, local_folder, is_data=True, file_name='AnalysisResults.root'):
    if isinstance(runs, (int, float, str)):
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
