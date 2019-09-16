#!/usr/bin/env python
import subprocess
import numpy as np
import dhfcorr.definitions as definitions


def get_train_output_file_name(folder_path, file_name):
    command = 'alien_find ' + folder_path + ' ' + file_name
    files = subprocess.run(command.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
    files = files.split('\n')[:-3]
    files = [x.strip() for x in files]
    return files


def submit_download_job(name, user_, password_, grid, local):
    commands = 'qsub -V -cwd -N ' + name + ' ' + definitions.ROOT_DIR + '/io/download_file.sh' \
               + ' ' + definitions.ROOT_DIR + ' ' + user_ + ' ' + password_ + ' ' + grid + ' ' + local
    subprocess.run(commands.split())


def get_friendly_file_name(file_name, train_name):
    list_path = file_name.split('/')
    part_with_child_name = list_path[list_path.index(train_name) + 1]
    id_file = list_path[list_path.index(train_name) + 2]

    return 'child' + part_with_child_name.split('child')[1] + '_' + id_file


def get_train_name_from_path(path):
    list_path = path.split('/')
    return list_path[list_path.index([x for x in list_path if x.startswith('PWG')][0]) + 1]


def get_period_from_path(path):
    list_path = path.split('/')
    period = list_path[4]
    short_name = period[3:]

    if not period.startswith('LHC'):
        raise (ValueError, "The period does not with LHC. Likely it was not produced with "
                           "the derived dataset. The code cannot handle this case")
    return short_name


def download_train_opt(folder_path_alien, local_folder,
                       user_, password_, n_batches=100,
                       file_name='AnalysisResults.root'):
    print("The following alien path was given: " + folder_path_alien)
    files = get_train_output_file_name(folder_path_alien, file_name)
    period_base = get_period_from_path(folder_path_alien)
    train_name = get_train_name_from_path(folder_path_alien)

    print("From which I could deduce that:")
    print("Train name: " + train_name)
    print("Period name: " + period_base + "\n")

    print('Number of files in this path:' + str(len(files)))

    friendly_names = [get_friendly_file_name(x, train_name) for x in files]

    def batch(sequence, n=1):
        length = len(sequence)
        for ndx in range(0, length, n):
            yield sequence[ndx:min(ndx + n, length)]

    current_job = 0
    size_jobs = len(files) / n_batches
    print("Number of jobs that will be submitted (approx.): " + str(size_jobs))

    for grid_file, local_file in zip(batch(files, n_batches), batch(friendly_names, n_batches)):
        local_file = [local_folder + '/' + period_base + '_' + x + '.root' for x in local_file]
        name_job = period_base + '_' + str(current_job)

        print("Submitting job " + str(name_job))
        submit_download_job('down_' + name_job, user_, password_,
                            format_list_to_bash(grid_file), format_list_to_bash(local_file))
        current_job = current_job + 1

    return size_jobs


def format_list_to_bash(string_list):
    merged = ''
    for x in string_list:
        merged += x + ','
    return merged[:-1]


if __name__ == '__main__':
    print("Downloading the output from the LEGO trains \n\n")
    print("                 o  o  O  O\n"
          "            ,_____  ____    O\n"
          "            | LHC \_|[]|_'__Y\n"
          "            |_______|__|_|__|}\n"
          "=============oo--oo==oo--OOO\\====================\n\n")

    import sys
    import dhfcorr.definitions as definitions
    import pandas as pd

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("user", help='User on grid')
    parser.add_argument("password", help='Password of the certificate in the .globus folder')
    parser.add_argument("txt_file", help='Text file (must end with .txt) with a list of paths to download. \n'
                                         'If the end is not .txt, it is assumed to be the path on grid.')
    parser.add_argument("destination", help="Destination of the file that will be downloaded.")

    args = parser.parse_args()
    user = args.user
    password = args.password
    txt_file = args.txt_file
    local_folder_path = definitions.DATA_FOLDER + '/' + str(args.destination)

    print('The files will be save to: ' + local_folder_path)

    if txt_file.endswith('.txt'):
        folders = list(pd.read_csv(txt_file, header=None)[0].values)
        print("A list of folders was given in the file " + str(txt_file) + ". They are the following:")
        for text in folders:
            print(text)
        print()

        for folder, i in zip(folders, range(len(folders))):
            print('-> -> -> -> -> ->- > -> -> -> -> -> -> -> ->- > -> ->')
            print("        Downloading folder        " + str(i + 1) + '/' + str(len(folders)))
            print('-> -> -> -> -> ->- > -> -> -> -> -> -> -> ->- > -> ->')
            size = download_train_opt(folder, local_folder_path, user, password)

    else:
        download_train_opt(txt_file, local_folder_path)
