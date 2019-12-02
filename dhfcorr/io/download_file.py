#!/usr/bin/env python
import pandas as pd
import subprocess
import time
import os


def get_token(code_token):
    token = subprocess.Popen('echo ' + code_token + ' | jalien', shell=True, stdout=subprocess.PIPE)
    token.wait()
    time.sleep(3)


def download_file_list(grid_path, local_path):
    for grid, local in zip(grid_path, local_path):
        download_file(grid, local)


def download_file(grid_path, local_path):
    command = 'echo cp  ' + str(grid_path) + ' file:' + local_path + ' | jalien'
    download = subprocess.Popen(command, shell=True)
    download.wait()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("login", help='CSV file used to login')
    parser.add_argument("csv_file", help='CSV file with locations from grid and local')

    args = parser.parse_args()
    login = pd.read_csv(args.login).iloc[0]
    user = login['user']
    code = login['code']
    # token = subprocess.Popen('echo ' + code + ' | alien-token-init ' + user, shell=True,
    #                         stdout=subprocess.PIPE)
    # token.wait()

    get_token(code)

    files = pd.read_csv(args.csv_file)
    download_file_list(files['grid'], files['local'])

    #os.remove(args.csv_file)
