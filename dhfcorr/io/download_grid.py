#!/usr/bin/env python
import subprocess
import warnings


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


def get_train_name_from_path(path):
    list_path = path.split('/')
    return list_path[list_path.index([x for x in list_path if x.startswith('PWG')][0]) + 1]


def get_period_from_path(path):
    list_path = path.split('/')
    period = list_path[4]
    short_name = period[3:]

    if not period.startswith('LHC'):
        warnings.warn(
            "The period does not with LHC. Likely it was not produced with the derived dataset. Using train name instead")
        folders = list_path
        train_number = folders[folders.index([x for x in folders if x.startswith('PWG')][0]) + 2].split('_')[0]
        short_name = train_number

    return short_name


def download_train_opt(folder_path_alien, local_folder, file_name='AnalysisResults.root'):
    print("Downloading the output from the LEGO trains \n\n")
    print("                 o  o  O  O\n"
          "            ,_____  ____    O\n"
          "            | LHC \_|[]|_'__Y\n"
          "            |_______|__|_|__|}\n"
          "=============oo--oo==oo--OOO\\====================\n\n")

    print("The following alien path was given: " + alien_path + "\n")
    files = get_train_output_file_name(folder_path_alien, file_name)
    period_base = get_period_from_path(folder_path_alien)
    train_name = get_train_name_from_path(folder_path_alien)

    print("From which I could deduce that: \n")
    print("Train name: " + period_base + "\n")
    print("Period name: " + period_base + "\n")
    print()

    print('I found the following files in this path:')

    for x in files:
        print(x)

    friendly_names = [get_friendly_file_name(x, train_name) for x in files]

    print()
    print('Saving files to local directory: ' + local_folder)
    print('File on GRID -----> File on local system')
    for grid_file, local_file in zip(files, friendly_names):
        local_file = period_base + '_' + local_file + '.root'
        print(grid_file + ' -----> ' + local_file)
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


if __name__ == '__main__':
    """"Download the output from the trains. Be sure to use the option 'generate devired dataset' or that the output of 
    the train was fully merged.
    
    The form of the command should be:
    
    download_grid.py folder_path_alien local_folder 
    """

    import sys
    import dhfcorr.definitions as definitions

    alien_path = str(sys.argv[1])
    try:
        local_folder_path = str(sys.argv[2])
    except IndexError:
        local_folder_path = definitions.ROOT_DIR

    download_train_opt(alien_path, local_folder_path)
