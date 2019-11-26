#!/usr/bin/env python

import subprocess
import os
from dhfcorr.submit_job import get_job_command
import dhfcorr.definitions as definitions
import pandas as pd
import itertools
import glob
from tqdm import tqdm

"""Tools to download from GRID to local cluster. 
The cluster must have a working copy of alien (alien_cp and alien_token-init are used).

In order to download the output, you will have to provide where the folders are saved. This can be obtained using:
1. Go to the train run and click on the "processing progress" of any "child". On the new page, it will show you the 
output folder of this 'child'. 
2. Edit the filter option for the output directory. This is the field between "Software versions" and "Job states".
Then remove all the content after the train rum number. You should change from something like 
"PWGHF/HFCJ_pp/561_20190909-1306_child_1$" to "PWGHF/HFCJ_pp/561".
3. Now copy all the content from the "Output directory" to a text file. You can use Excel to help you select only the 
correct columns.
4. Download the files using python dhfcorr/io/submit_download_grid.py grid_username certificate_password location_txt_file 
destination_folder
Check submit_download_grid.py --help to a description of each parameter.

"""


def get_train_output_file_name(folder_path, file_name):
    command = 'alien_find ' + folder_path + ' ' + file_name
    files = subprocess.run(command.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
    files = files.split('\n')[:-3]
    files = [x.strip() for x in files]
    return files


def submit_download_job(name, login, csv_file):
    commands = get_job_command(name, definitions.ROOT_DIR + '/io/download_file.py',
                               login + ' ' + csv_file)
    subprocess.run(commands, shell=True)


def get_friendly_file_name(file_name, train_name):
    list_path = file_name.split('/')
    period = str(get_period_from_path(file_name))
    run_number = get_run_number_from_path(file_name)
    id_file = list_path[list_path.index(train_name) + 2]
    return period + '_' + run_number + '_' + id_file


def get_train_name_from_path(path):
    list_path = path.split('/')
    return list_path[list_path.index([x for x in list_path if x.startswith('PWG')][0]) + 1]


def get_period_from_path(path):
    list_path = path.split('/')
    period = list_path[4]
    short_name = period[3:]

    if not period.startswith('LHC'):
        raise (ValueError, "The period does not with LHC. Likely it was not produced with "
                           "the derived dataset option. The package cannot handle this case.")
    return short_name


def get_run_number_from_path(path):
    list_path = path.split('/')
    return list_path[5]


def download_train_opt(train_name, run_number, local_folder,
                       login, n_batches=100,
                       file_name='AnalysisResults.root'):
    files = find_files_from_train(run_number, train_name, file_name=file_name)
    files_downloaded = glob.glob(local_folder + '/*.root')

    print("Train name: " + train_name)
    print("Train run number: " + run_number)

    print('Number of files in this path: ' + str(len(files)))
    friendly_names = [get_friendly_file_name(x, train_name) for x in files]
    friendly_names_downloaded = [x.split('/')[-1].split('.root')[0] for x in files_downloaded]
    print('Number of files in already downloaded path: ' + str(len(friendly_names_downloaded)))
    files = [files[x] for x in range(len(files))
             if friendly_names[x] not in friendly_names_downloaded]
    friendly_names = [friendly_names[x] for x in range(len(files))
                      if friendly_names[x] not in friendly_names_downloaded]
    print(len(files))
    print(len(friendly_names))
    print('So I will download:' + str(len(files)))
    size_jobs = len(files) / n_batches
    print("Number of jobs that will be submitted (approx.): " + str(size_jobs))

    from dhfcorr.io.utils import batch

    current_job = 0
    for grid_file, local_file in zip(batch(files, n_batches), batch(friendly_names, n_batches)):
        local_file = [local_folder + '/' + x + '.root' for x in local_file]
        name_job = 'down_' + str(run_number) + str('_') + str(current_job)

        print("Saving grid files and destinations on csv file: ", name_job + '.csv')

        pd.DataFrame({'grid': grid_file, 'local': local_file}).to_csv(name_job + '.csv')

        print("Submitting job " + str(name_job))

        submit_download_job(name_job, login, os.path.join(os.getcwd(), name_job + '.csv'))
        current_job = current_job + 1


def get_files_from_alien_opt(run_result):
    files = run_result.split('\n')[:-3]
    files = [x.strip() for x in files]
    return files


def find_files_on_grid(folder, file_pattern):
    command = 'alien_find ' + folder + ' ' + file_pattern
    res = get_files_from_alien_opt(subprocess.run(command, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8'))
    return res


def download_from_grid(location, destination):
    download = subprocess.Popen('alien_cp  alien://' + location + ' ' + destination, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, shell=True)
    download.wait()


def find_output_dirs(env_file):
    output_dirs = list()
    with open(env_file) as f:
        for line in f:
            for word in line.split():
                if "OUTPUTDIR" in word:
                    dir_data = '/' + '/'.join(word.split('=')[1].split('/')[1:5]) + '/'
                    output_dirs.append(dir_data)
    return output_dirs


def find_files_from_train(train_run, train_name, pwg='PWGHF', file_name='AnalysisResults.root'):
    # Find the env file of the train
    env_folder = "/alice/cern.ch/user/a/alitrain/" + str(pwg) + "/" + str(train_name) + "/"
    env_file_pattern = str(train_run) + "_20*/env.sh"

    env_file = find_files_on_grid(env_folder, env_file_pattern)
    if len(env_file) < 1:
        message = "The env file was not found on the GRID. Please check the train name and train run."
        message += "\n The train_run is " + str(train_run) + " and train name is " + str(train_name)
        message += " and PWG is " + str(pwg)
        raise FileNotFoundError(message)
    print("Retriving LEGO train configuration from:")
    print(env_file[0])
    download_from_grid(env_file[0], 'temp_env.sh')
    output_dirs = find_output_dirs('temp_env.sh')

    print("The following datasets were found: ")
    for x in output_dirs:
        print(x)

    os.remove('temp_env.sh')

    files_to_download = list()
    print("Building file list to be downloaded:")

    for folder in tqdm(output_dirs):
        files_path = '*/' + train_name + '/' + str(train_run) + '*/' + file_name
        files = find_files_on_grid(folder, files_path)
        files_to_download.append(files)

    files_on_grid = list(itertools.chain.from_iterable(files_to_download))

    return files_on_grid


if __name__ == '__main__':
    print("Downloading the output from the LEGO trains \n\n")
    # image from https://www.asciiart.eu/vehicles/trains
    print("                 o  o  O  O\n"
          "            ,_____  ____    O\n"
          "            | LHC \_|[]|_'__Y\n"
          "            |_______|__|_|__|}\n"
          "=============oo--oo==oo--OOO\\====================\n\n")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("user", help='User on grid')
    parser.add_argument("code", help='Code to unlock the certificate')
    parser.add_argument("train_name", help='Name of the train (eg. HFCJ_pp')
    parser.add_argument("destination", help="Destination of the file that will be downloaded. It is always added to "
                                            "the basic definitions from the definitions.py file")
    parser.add_argument("-r", "--train_runs", help='Number of the run in the Lego train system', nargs='+',
                        required=True)

    parser.add_argument('-c', '--continue_previous', dest='continue_down', action='store_true')
    parser.add_argument('-d', '--delete_previous', dest='continue_down', action='store_false')
    parser.set_defaults(continue_down=True)

    parser.add_argument("-n", "--n_files", help='Number of files in each job ', default=50)

    args = parser.parse_args()

    local_folder_path = definitions.DATA_FOLDER + '/root/' + str(args.destination)
    pd.DataFrame([{'user': args.user, 'code': args.code}]).to_csv('~/login.csv')

    if not os.path.isdir(definitions.DATA_FOLDER + '/root/'):
        os.mkdir(definitions.DATA_FOLDER + '/root/')

    if not os.path.isdir(local_folder_path):
        os.mkdir(local_folder_path)

    print('The files will be save to: ' + local_folder_path)

    print("The train run numbers are:")
    print(args.train_runs)

    for run, i in zip(args.train_runs, range(len(args.train_runs))):
        print('-> -> -> -> -> ->- > -> -> -> -> -> -> -> ->- > -> ->')
        print("         Downloading train run: " + str(run))
        print('-> -> -> -> -> ->- > -> -> -> -> -> -> -> ->- > -> ->')
        download_train_opt(args.train_name, run, local_folder_path, '~/login.csv')
